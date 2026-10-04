"""Regression guard for the rule layer: every decision it makes on the probe set must stay right."""

from text2sql.eval.rules import evaluate_rules, load_probes
from text2sql.pipeline.router import Router
from text2sql.semantic.registry import load_semantic_model

reports = evaluate_rules(Router(load_semantic_model(), llm=None), load_probes())


def test_metadata_rule_never_misfires():
    assert reports["META"].wrong == []
    assert reports["META"].decided == 10  # the other 6 phrasings are left to the model


def test_out_of_domain_rule_misfires_only_on_the_known_vague_question():
    assert reports["OOD"].wrong == ["今年整体怎么样？"]  # no domain word, no data word: accepted limitation
    assert reports["OOD"].decided == 8
