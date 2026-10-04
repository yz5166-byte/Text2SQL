"""Report the rule layer's precision and coverage on eval/rule_probes.yaml (no model needed)."""

from text2sql.eval.rules import evaluate_rules, load_probes
from text2sql.pipeline.router import Router
from text2sql.semantic.registry import load_semantic_model


def main() -> None:
    reports = evaluate_rules(Router(load_semantic_model(), llm=None), load_probes())
    for r in reports.values():
        made = r.decided + len(r.wrong)
        print(f"{r.label}: rules decided {made}, correct {r.decided} "
              f"(precision {r.decided}/{made}); coverage {r.decided}/{r.total}")
        for q in r.wrong:
            print(f"  wrong   {q}")
        for q in r.abstained:
            print(f"  to model {q}")


if __name__ == "__main__":
    main()
