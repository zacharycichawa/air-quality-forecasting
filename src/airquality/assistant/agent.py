"""Tool-using assistant: answers questions about Polish air quality with three tools.

    run_sql            read-only SQL over the `bi` views (text-to-SQL)
    get_forecast       latest 24/48 h PM2.5 forecasts for a city
    search_guidelines  retrieval over WHO/EU standards, health info, methodology (RAG)
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import Engine, text

from airquality.assistant.knowledge import KnowledgeIndex
from airquality.assistant.llm import OllamaClient
from airquality.assistant.sql_tool import UnsafeQuery, run_select

MAX_STEPS = 6
MAX_TOOL_CHARS = 6000

TOOLS = [
    {"type": "function", "function": {
        "name": "run_sql",
        "description": "Run ONE read-only PostgreSQL SELECT query over the bi views and return "
                       "the rows. Use for any question about measured air quality data.",
        "parameters": {"type": "object", "required": ["query"], "properties": {
            "query": {"type": "string", "description": "A single SELECT statement."}}},
    }},
    {"type": "function", "function": {
        "name": "get_forecast",
        "description": "Get the latest PM2.5 forecast values (24 h and 48 h ahead, plus the "
                       "chance of PM2.5 above 50 µg/m³) for the stations in a city the user "
                       "names. Not for questions about how accurate the forecasts are.",
        "parameters": {"type": "object", "required": ["city"], "properties": {
            "city": {"type": "string", "description": "City name, e.g. Kraków or Łódź."}}},
    }},
    {"type": "function", "function": {
        "name": "search_guidelines",
        "description": "Search reference documents: WHO 2021 air quality guidelines, EU legal "
                       "limits, health effects and advice, sources of smog in Poland, and how "
                       "this project's data and forecasts work, including the forecast "
                       "model's accuracy and evaluation results.",
        "parameters": {"type": "object", "required": ["query"], "properties": {
            "query": {"type": "string", "description": "What to look up."}}},
    }},
]

SCHEMA_NOTES = """
Views (schema bi). All concentrations are in µg/m³.
- bi.fact_daily_air_quality: one row per station, pollutant and day (valid daily means only,
  >= 18 hourly values). Columns: station_code, pollutant ('PM2.5' | 'PM10' | 'NO2'),
  date (Polish local date), mean_ug_m3, max_hourly_ug_m3, n_hours,
  exceeds_who_daily (boolean), exceeds_eu_daily (boolean, PM10 only, EU limit 50).
- bi.dim_station: station_code, station_name, city, voivodeship, latitude, longitude,
  is_mobile (temporary mobile stations), measures_pm25.
- bi.dim_date: date, year, quarter, month, month_name, year_month, day_of_week (1 = Monday),
  day_name, is_weekend, is_heating_season (October-March). These calendar columns exist
  ONLY here: JOIN bi.dim_date USING (date) to use them.
- bi.dim_threshold: pollutant, standard ('WHO 2021 guideline' | 'EU limit'), averaging
  ('daily' | 'annual'), threshold_ug_m3, note.
- bi.fact_forecast: model ('lightgbm' | 'persistence'), run_type ('live' | 'backtest'),
  station_code, horizon_h (24 | 48), issue_local, target_local, target_date,
  predicted_ug_m3, actual_ug_m3, abs_error, p_episode (probability of PM2.5 > 50).

SQL rules (important):
- ALWAYS filter on pollutant (pollutant = 'PM2.5' unless the question names PM10 or NO2).
  Mixing pollutants in one average gives a wrong number.
- Join facts to bi.dim_station on station_code to filter or group by city. A city can have
  several stations: for questions about cities, GROUP BY s.city (not the station).
- A city's or the country's average = AVG(mean_ug_m3) over the relevant daily rows.
- Years: EXTRACT(YEAR FROM date) or join bi.dim_date. Complete years: 2019-2025.
- Month ranges that wrap around the new year (e.g. October-March) can't use BETWEEN 10 AND 3
  (it matches nothing): use dim_date.is_heating_season or month IN (10, 11, 12, 1, 2, 3).
- If a query returns NULL or no rows where data is expected, assume the query is wrong and
  fix it. Never tell the user data is missing unless a correct query confirms it.
- When ranking cities or stations, exclude mobile stations (is_mobile = false).
- City names are stored with Polish characters, exactly as in the list below.
- Always add ORDER BY and LIMIT for rankings; round results to 1 decimal.

Example - average NO2 in Poznań in 2022:
SELECT ROUND(AVG(f.mean_ug_m3)::numeric, 1) FROM bi.fact_daily_air_quality f
JOIN bi.dim_station s USING (station_code)
WHERE f.pollutant = 'NO2' AND s.city = 'Poznań' AND EXTRACT(YEAR FROM f.date) = 2022;

Example - the 3 cities with the lowest PM10 in 2021:
SELECT s.city, ROUND(AVG(f.mean_ug_m3)::numeric, 1) AS pm10 FROM bi.fact_daily_air_quality f
JOIN bi.dim_station s USING (station_code)
WHERE f.pollutant = 'PM10' AND EXTRACT(YEAR FROM f.date) = 2021 AND NOT s.is_mobile
GROUP BY s.city ORDER BY pm10 ASC LIMIT 3;
"""

SYSTEM_TEMPLATE = """You are an assistant for a Polish air quality data project. Today is {today}.
Answer questions about air pollution in Poland using ONLY the tools' results.

Rules:
- For any number about measured data, call run_sql. Never guess or invent numbers.
- For forecast values for a named city ("tomorrow", "next days") call get_forecast.
  Never pick a city the user didn't mention.
- For standards, health effects, advice, causes or methodology - including how accurate
  or reliable the forecasts are - call search_guidelines and cite the source document
  titles (and URLs) you used.
- If a tool returns an error, fix your query and call the tool again. Never just show
  SQL to the user: always run it.
- If the tools can't answer, say so plainly. Decline requests unrelated to air quality
  and requests to change or delete data.
- Reply in the SAME language as the user's question: English question -> English answer,
  Polish question -> Polish answer. Be brief and clear, with units (µg/m³).
{schema}
Cities with PM2.5 stations: {cities}
"""


def _fold(s: str) -> str:
    """Lower-case and strip diacritics (Łódź -> lodz) for forgiving city matching."""
    s = s.replace("ł", "l").replace("Ł", "L")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
                   ).lower().strip()


SQL_BLOCK = re.compile(r"```sql\s*(.+?)```", re.DOTALL | re.IGNORECASE)


def _sql_in_text(text: str) -> list[dict]:
    """Recovery for a small-model failure mode: after a SQL error the model sometimes
    writes the corrected query in a ```sql block instead of calling the tool. Treat the
    last such block as a run_sql call (the same read-only guards apply)."""
    blocks = SQL_BLOCK.findall(text)
    return [{"function": {"name": "run_sql", "arguments": {"query": blocks[-1].strip()}},
             "recovered": True}] if blocks else []


@dataclass
class Answer:
    question: str
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    steps: int = 0


class Assistant:
    def __init__(self, reader: Engine, llm: OllamaClient | None = None,
                 index: KnowledgeIndex | None = None):
        self.db, self.llm = reader, llm or OllamaClient()
        self.index = index or KnowledgeIndex.load()
        self.cities = self._cities()

    def _cities(self) -> list[str]:
        rows = run_select(self.db, "SELECT DISTINCT city FROM bi.dim_station "
                                   "WHERE measures_pm25 ORDER BY city")["rows"]
        return [r[0] for r in rows]

    def system_prompt(self) -> str:
        return SYSTEM_TEMPLATE.format(today=date.today().isoformat(), schema=SCHEMA_NOTES,
                                      cities=", ".join(self.cities))

    # ---------------------------------------------------------------- tools
    def tool_run_sql(self, query: str) -> dict:
        try:
            return run_select(self.db, query)
        except UnsafeQuery as exc:
            return {"error": f"query rejected: {exc}"}
        except Exception as exc:  # database errors are fed back so the model can fix the SQL
            return {"error": str(getattr(exc, "orig", exc)).split("\n")[0][:300]}

    def tool_get_forecast(self, city: str) -> dict:
        matches = [c for c in self.cities if _fold(c) == _fold(city)] or \
                  [c for c in self.cities if _fold(city) in _fold(c)]
        if not matches:
            return {"error": f"no PM2.5 stations found for city '{city}'"}
        with self.db.connect() as conn:
            rows = conn.execute(text("""
                SELECT s.city, s.station_name, f.horizon_h, f.target_local, f.uses_weather,
                       round(f.predicted_ug_m3::numeric, 1) AS predicted_ug_m3,
                       round(f.p_episode::numeric, 3) AS chance_above_50, f.issue_local
                  FROM bi.fact_forecast f JOIN bi.dim_station s USING (station_code)
                 WHERE f.run_type = 'live' AND f.model = 'lightgbm' AND s.city = ANY(:cities)
                   AND f.issue_local = (SELECT max(issue_local) FROM bi.fact_forecast g
                                         WHERE g.run_type = 'live'
                                           AND g.station_code = f.station_code)
                 ORDER BY s.station_name, f.horizon_h"""), {"cities": matches}).mappings().all()
        if not rows:
            return {"error": f"no current forecast for {', '.join(matches)} (stations may "
                             f"not have reported recently)"}
        uses_weather = any(r["uses_weather"] for r in rows)
        return {"forecasts": [dict(r) for r in rows],
                "note": ("model uses pollution history, calendar and the weather forecast"
                         if uses_weather else "model uses pollution history and calendar")}

    def tool_search_guidelines(self, query: str) -> dict:
        hits = self.index.search(self.llm.embed([query])[0], k=3)
        return {"passages": [{"document": c.doc_title, "section": c.section, "url": c.url,
                              "text": c.text, "score": round(s, 3)} for c, s in hits]}

    def _call(self, name: str, args: dict) -> dict:
        fn = {"run_sql": self.tool_run_sql, "get_forecast": self.tool_get_forecast,
              "search_guidelines": self.tool_search_guidelines}.get(name)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except TypeError as exc:
            return {"error": f"bad arguments for {name}: {exc}"}

    # ---------------------------------------------------------------- loop
    def ask(self, question: str) -> Answer:
        t0 = time.time()
        messages = [{"role": "system", "content": self.system_prompt()},
                    {"role": "user", "content": question}]
        trace: list[dict] = []
        for step in range(1, MAX_STEPS + 1):
            msg = self.llm.chat(messages, TOOLS)
            messages.append({"role": "assistant", "content": msg.get("content", "")})
            calls = msg.get("tool_calls") or _sql_in_text(msg.get("text", ""))
            if not calls:
                answer = msg.get("text", msg.get("content", "")).strip()
                return Answer(question, answer, trace, round(time.time() - t0, 1), step)
            for call in calls:
                name = call["function"]["name"]
                args = call["function"].get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args or "{}")
                result = self._call(name, args)
                trace.append({"tool": name, "args": args,
                              "error": result.get("error") if isinstance(result, dict) else None,
                              "recovered": call.get("recovered", False)})
                content = json.dumps(result, default=str, ensure_ascii=False)[:MAX_TOOL_CHARS]
                messages.append({"role": "tool", "content": content, "tool_name": name})
        return Answer(question, "Sorry, I couldn't complete that within the step limit.",
                      trace, round(time.time() - t0, 1), MAX_STEPS)
