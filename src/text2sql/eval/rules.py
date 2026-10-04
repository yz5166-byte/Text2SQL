"""Evaluate the rule layer on probe questions (eval/rule_probes.yaml).

A rule may abstain, leaving the question to the model; that costs one model call but is not an
error. A decision the rules do make must be right: META and OOD probes must get exactly that
route, and DATA probes must not be routed to META or OOD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from text2sql.config import EVAL_DIR
from text2sql.llm.base import LLMError
from text2sql.pipeline.router import Router


@dataclass
class LabelReport:
    label: str
    total: int = 0
    decided: int = 0  # probes of this label the rules decided by themselves
    wrong: list[str] = field(default_factory=list)  # questions the rules sent to this route wrongly
    abstained: list[str] = field(default_factory=list)  # probes of this label left to the model


def load_probes(path: Path = EVAL_DIR / "rule_probes.yaml") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["probes"]


def evaluate_rules(router: Router, probes: list[dict]) -> dict[str, LabelReport]:
    """Per label (META, OOD): how often the rules decided it, and every wrong decision."""
    reports = {label: LabelReport(label) for label in ("META", "OOD")}
    for probe in probes:
        try:
            route = router.route(probe["q"]).route
        except LLMError:
            route = None  # no rule applied and no model configured: left to the model
        expect = probe["expect"]
        if expect in reports:
            reports[expect].total += 1
            if route == expect:
                reports[expect].decided += 1
            else:
                reports[expect].abstained.append(probe["q"])
        if route in reports and route != expect:
            reports[route].wrong.append(probe["q"])
    return reports
