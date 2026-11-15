"""Assistant agent loop (scripted fake LLM) and evaluation scoring - no Ollama needed."""

from __future__ import annotations

import numpy as np
import pytest

from airquality.assistant.agent import Answer, Assistant, _fold
from airquality.assistant.evaluate import numbers_in, score
from airquality.assistant.knowledge import Chunk, KnowledgeIndex


class FakeLLM:
    """Replays a fixed sequence of assistant messages and records what it was sent."""

    model = "fake"

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def chat(self, messages, tools=None):
        self.seen.append([dict(m) for m in messages])
        return self.replies.pop(0)

    def embed(self, texts):
        return np.ones((len(texts), 2))


def call(name, **args):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": args}}]}


@pytest.fixture
def assistant(monkeypatch):
    index = KnowledgeIndex([Chunk("WHO AQG", "PM", "WHO", "https://who.int", "PM2.5 5 µg/m³")],
                           np.ones((1, 2)))
    monkeypatch.setattr(Assistant, "_cities", lambda self: ["Kraków", "Łódź"])

    def make(replies):
        a = Assistant(reader=None, llm=FakeLLM(replies), index=index)
        a.tool_run_sql = lambda query: ({"error": "query rejected: only SELECT"}
                                        if query.startswith("DROP") else {"rows": [[15.5]]})
        return a
    return make


def test_loop_feeds_tool_errors_back_and_returns_final_answer(assistant):
    a = assistant([call("run_sql", query="DROP TABLE x"),
                   call("run_sql", query="SELECT 15.5"),
                   {"role": "assistant", "content": "The average was 15.5 µg/m³."}])
    ans = a.ask("average?")
    assert ans.answer == "The average was 15.5 µg/m³." and ans.steps == 3
    assert [t["error"] for t in ans.tool_calls] == ["query rejected: only SELECT", None]
    second_turn = a.llm.seen[1]
    assert second_turn[-1]["role"] == "tool" and "rejected" in second_turn[-1]["content"]


def test_search_tool_returns_cited_passages(assistant):
    a = assistant([call("search_guidelines", query="WHO PM2.5"),
                   {"role": "assistant", "content": "5 µg/m³ (WHO AQG)."}])
    a.ask("WHO guideline?")
    tool_msg = a.llm.seen[1][-1]["content"]
    assert "https://who.int" in tool_msg and "WHO AQG" in tool_msg


def test_unknown_tool_and_step_limit(assistant):
    a = assistant([call("nope")] * 6)
    ans = a.ask("loop forever")
    assert ans.steps == 6 and "step limit" in ans.answer
    assert ans.tool_calls[0]["error"] == "unknown tool nope"


def test_system_prompt_lists_cities_with_polish_characters(assistant):
    assert "Kraków, Łódź" in assistant([]).system_prompt()


def test_fold_and_numbers():
    assert _fold("Łódź") == "lodz" and _fold("KRAKÓW ") == "krakow"
    assert numbers_in("11,475 of 32,348 days; 0,355") == [11475.0, 32348.0, 0.355]
    assert numbers_in("Średnia: 15,5 µg/m³, a w 2019 r. 24.8; razem 1 234") == [
        15.5, 2019.0, 24.8, 1234.0]


@pytest.mark.parametrize("check, truth, text, expected", [
    ("number", [15.51], "It was about 15.5 µg/m³.", True),
    ("number", [15.51], "It was 17 µg/m³.", False),
    ("percent", [35.47], "About 35% of days.", True),    # within 2 % of truth (0.71)
    ("percent", [35.47], "About 31% of days.", False),
    ("percent", [35.47], "Roughly 35.5 % of days.", True),
    ("percent", [35.47], "A share of 0.355.", True),
    ("label", ["WIELKOPOLSKIE"], "Wielkopolskie had the highest level.", True),
    ("month", [2], "Najwyższe stężenia są w lutym... luty", True),
])
def test_score_checks(check, truth, text, expected):
    ok, _ = score({"check": check}, Answer("q", text), truth, {})
    assert ok is expected


def test_parse_tool_calls_accepts_multiline_sql():
    from airquality.assistant.llm import parse_tool_calls, with_tools

    raw = ('<tool_call>\n{"name": "run_sql", "arguments": {"query": "SELECT city\nFROM '
           'bi.dim_station\nLIMIT 1;"}}\n</tool_call>')
    text, calls = parse_tool_calls(raw)
    assert text == "" and calls[0]["function"]["name"] == "run_sql"
    assert calls[0]["function"]["arguments"]["query"].startswith("SELECT city\nFROM")
    assert parse_tool_calls("Plain answer.") == ("Plain answer.", [])
    msgs = with_tools([{"role": "system", "content": "S"},
                       {"role": "assistant", "content": raw, "tool_calls": calls}],
                      [{"type": "function", "function": {"name": "run_sql"}}])
    assert "<tools>" in msgs[0]["content"] and "tool_calls" not in msgs[1]


def test_sql_written_as_text_is_recovered_and_executed(assistant):
    a = assistant([{"role": "assistant", "content": "Let me fix it:\n```sql\nSELECT 15.5\n```",
                    "text": "Let me fix it:\n```sql\nSELECT 15.5\n```", "tool_calls": []},
                   {"role": "assistant", "content": "It was 15.5 µg/m³.",
                    "text": "It was 15.5 µg/m³."}])
    ans = a.ask("average?")
    assert ans.answer == "It was 15.5 µg/m³."
    assert ans.tool_calls[0]["tool"] == "run_sql" and ans.tool_calls[0]["recovered"]
    assert ans.tool_calls[0]["args"]["query"] == "SELECT 15.5"
