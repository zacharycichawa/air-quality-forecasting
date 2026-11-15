"""Knowledge-base chunking and retrieval (with a fake, deterministic embedder)."""

from __future__ import annotations

import numpy as np

from airquality.assistant.knowledge import KnowledgeIndex, chunk_markdown, load_corpus

DOC = """---
title: Test doc
source: Somewhere
url: https://example.org
---

## First section

Alpha   beta.

## Second section

Gamma delta.
"""


def test_chunks_carry_title_section_and_source():
    chunks = chunk_markdown(DOC)
    assert [c.section for c in chunks] == ["First section", "Second section"]
    assert chunks[0].text == "Alpha beta." and chunks[0].url == "https://example.org"
    assert chunks[1].for_embedding().startswith("Test doc - Second section")


def test_real_corpus_loads_with_sources():
    chunks = load_corpus()
    assert len(chunks) >= 15
    assert all(c.source and c.url for c in chunks)


def fake_embed(texts):
    vocab = ["alpha", "gamma", "pm2.5", "eu"]
    return np.array([[t.lower().count(w) + 1e-3 for w in vocab] for t in texts], dtype=float)


def test_search_returns_most_similar_chunk(tmp_path):
    (tmp_path / "doc.md").write_text(DOC, encoding="utf-8")
    index = KnowledgeIndex.build(fake_embed, tmp_path)
    best, score = index.search(fake_embed(["gamma please"])[0], k=1)[0]
    assert best.section == "Second section" and score > 0.9
    index.save(tmp_path / "idx")
    again = KnowledgeIndex.load(tmp_path / "idx")
    assert again.search(fake_embed(["alpha"])[0], k=1)[0][0].section == "First section"
