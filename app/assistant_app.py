"""Local chat UI for the air quality assistant (needs PostgreSQL + Ollama running).

Run:  streamlit run app/assistant_app.py
Unlike app/streamlit_app.py this is not deployable as-is: it talks to the local database
(through the read-only role) and to the local LLM.
"""

from __future__ import annotations

import json

import streamlit as st

from airquality.assistant.agent import Assistant
from airquality.db import get_engine

st.set_page_config(page_title="Air Quality Assistant", page_icon="💬", layout="centered")
st.title("Ask about Polish air quality")
st.caption("A local LLM (via Ollama) answers using three tools: read-only SQL over the "
           "project database, the latest PM2.5 forecasts, and a search over WHO/EU guidelines. "
           "Expand *How I got this* under an answer to see the exact tool calls.")

EXAMPLES = ["Which city had the highest average PM2.5 in 2024?",
            "What is the WHO annual guideline for PM2.5?",
            "What PM2.5 level is forecast for Łódź tomorrow?",
            "Jak zmieniło się stężenie PM2.5 w Krakowie od 2019 roku?"]


@st.cache_resource
def assistant() -> Assistant:
    return Assistant(get_engine(reader=True))


if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        with st.expander(f"How I got this ({turn['seconds']} s)"):
            for c in turn["tool_calls"]:
                st.markdown(f"**{c['tool']}**" + (f" (error: {c['error']})" if c["error"] else ""))
                if c["tool"] == "run_sql":
                    st.code(c["args"].get("query", ""), language="sql")
                else:
                    st.code(json.dumps(c["args"], ensure_ascii=False), language="json")

cols = st.columns(2)
clicked = next((q for i, q in enumerate(EXAMPLES) if cols[i % 2].button(q, key=f"ex{i}")), None)
question = st.chat_input("Ask a question in English or Polish") or clicked

if question:
    with st.spinner("Thinking (a local model can take a while)…"):
        ans = assistant().ask(question)
    st.session_state.history.append({"question": question, "answer": ans.answer,
                                     "tool_calls": ans.tool_calls, "seconds": ans.seconds})
    st.rerun()
