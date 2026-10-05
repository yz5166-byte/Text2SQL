"""Few-shot SQL examples for layer 3: a hand-written library, picked by similarity to the question."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from text2sql.config import ROOT
from text2sql.knowledge.retriever import HybridIndex, Method

EXAMPLES_PATH = ROOT / "knowledge" / "sql_examples.yaml"


@dataclass(frozen=True)
class SQLExample:
    id: str
    question: str
    sql: str
    explanation: str


def load_examples(path: Path = EXAMPLES_PATH) -> list[SQLExample]:
    with open(path, encoding="utf-8") as f:
        return [SQLExample(**e) for e in yaml.safe_load(f)["examples"]]


class ExamplePicker:
    def __init__(self, examples: list[SQLExample], method: Method = "hybrid"):
        self.examples = {e.id: e for e in examples}
        self.index = HybridIndex({e.id: e.question for e in examples}, method)

    def pick(self, question: str, k: int = 3) -> list[SQLExample]:
        return [self.examples[i] for i in self.index.rank(question)[:k]]
