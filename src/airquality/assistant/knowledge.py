"""Retrieval over the small knowledge base in docs/knowledge/ (the RAG part).

The corpus is a few dozen passages, so a NumPy matrix + cosine similarity is all that
is needed; a vector database would add infrastructure without any benefit at this size.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from airquality.config import DATA_DIR, KNOWLEDGE_DIR

INDEX_DIR = DATA_DIR / "assistant"
Embedder = Callable[[list[str]], np.ndarray]


@dataclass(frozen=True)
class Chunk:
    doc_title: str
    section: str
    source: str
    url: str
    text: str

    def for_embedding(self) -> str:
        return f"{self.doc_title} - {self.section}\n{self.text}"


def _front_matter(md: str) -> tuple[dict, str]:
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", md, flags=re.DOTALL)
    if not m:
        return {}, md
    meta = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
    return {k.strip(): v.strip() for k, v in meta.items()}, m.group(2)


def chunk_markdown(md: str) -> list[Chunk]:
    """One chunk per '## ' section, carrying the document's title and source."""
    meta, body = _front_matter(md)
    chunks = []
    for part in re.split(r"(?m)^## ", body)[1:]:
        heading, _, text = part.partition("\n")
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            chunks.append(Chunk(meta.get("title", ""), heading.strip(), meta.get("source", ""),
                                meta.get("url", ""), text))
    return chunks


def load_corpus(directory: Path = KNOWLEDGE_DIR) -> list[Chunk]:
    return [c for p in sorted(directory.glob("*.md"))
            for c in chunk_markdown(p.read_text(encoding="utf-8"))]


def _normalise(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


class KnowledgeIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        self.chunks, self.vectors = chunks, _normalise(vectors.astype("float32"))

    @classmethod
    def build(cls, embed: Embedder, directory: Path = KNOWLEDGE_DIR) -> KnowledgeIndex:
        chunks = load_corpus(directory)
        return cls(chunks, embed([c.for_embedding() for c in chunks]))

    def search(self, query_vector: np.ndarray, k: int = 3) -> list[tuple[Chunk, float]]:
        q = _normalise(query_vector.reshape(1, -1).astype("float32"))[0]
        scores = self.vectors @ q
        best = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in best]

    def save(self, directory: Path = INDEX_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        (directory / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False, indent=1),
            encoding="utf-8")

    @classmethod
    def load(cls, directory: Path = INDEX_DIR) -> KnowledgeIndex:
        chunks = [Chunk(**c) for c in json.loads(
            (directory / "chunks.json").read_text(encoding="utf-8"))]
        return cls(chunks, np.load(directory / "vectors.npy"))
