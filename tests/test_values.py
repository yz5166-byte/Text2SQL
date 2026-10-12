"""Value index: which stored values are indexed, whole-word and fuzzy matching, and the guards against accidental hits."""

import pytest

from text2sql.config import BIRD_DB_PATH
from text2sql.eval.dataset import bird_tables
from text2sql.knowledge.values import ValueIndex, normalize
from text2sql.semantic.registry import load_semantic_model

INDEX = ValueIndex({
    "card.type": ["gold", "classic", "junior"],
    "trans.operation": ["VYBER", "VYBER KARTOU", "VKLAD"],
    "trans.k_symbol": ["SIPO", "UROK", " "],
    "order.k_symbol": ["SIPO", "UVER"],
    "district.A2": ["Most", "Hl.m. Praha", "Usti nad Labem"],
    "district.A3": ["east Bohemia", "west Bohemia"],
    "district.A4": ["1204953", "88884"],
    "client.gender": ["F", "M"],
    "ads_okr_monthly.region": ["ALL"],
})


def hits(question):
    return {h.value: h for h in INDEX.match(question)}


def test_short_numeric_and_common_values_are_not_indexed():
    assert set(INDEX.entries) == {"gold", "classic", "junior", "vyber", "vyber kartou", "vklad", "sipo", "urok",
                                  "uver", "most", "hl.m. praha", "usti nad labem", "east bohemia", "west bohemia"}


def test_whole_words_without_case():
    found = hits("How many GOLD cards were issued in East Bohemia?")
    assert set(found) == {"gold", "east Bohemia"}
    assert found["east Bohemia"].columns == ("district.A3",) and found["gold"].surface == "GOLD"
    assert hits("goldsmiths and classical music") == {}  # inside a longer word: no match


def test_the_longest_value_wins_an_overlap():
    assert set(hits("withdrawals of type VYBER KARTOU")) == {"VYBER KARTOU"}
    assert set(hits("VYBER and VYBER KARTOU")) == {"VYBER", "VYBER KARTOU"}


def test_a_value_in_two_columns_names_both_tables():
    found = hits("payments for SIPO")["SIPO"]
    assert found.columns == ("trans.k_symbol", "order.k_symbol") and found.tables == ["trans", "order"]


def test_a_proper_name_must_be_capitalized():
    assert "Most" not in hits("Which district has the most accounts?")
    assert "Most" in hits("How many accounts were opened in Most?")


def test_a_short_code_must_match_exactly():
    index = ValueIndex({"drivers.code": ["BAD", "PER"], "disp.type": ["OWNER"]})
    assert {h.value for h in index.match("bad loans per month of the owner")} == {"OWNER"}
    assert {h.value for h in index.match("drivers with code BAD")} == {"BAD"}


def test_fuzzy_match_is_marked_and_needs_a_unique_best():
    found = hits("crimes in e ast bohemia in 1996")["east Bohemia"]
    assert found.fuzzy and found.surface == "ast bohemia"
    assert hits("clients in Usti nad Labemm")["Usti nad Labem"].fuzzy
    assert hits("crimes in est bohemia") == {}  # 0.8 to both east and west Bohemia: no unique best
    assert hits("crimes in Bohemia") == {}  # one word is compared only with one-word values


def test_normalize_drops_accents_and_extra_spaces():
    assert normalize("  Ústí  nad Labem ") == "usti nad labem"


@pytest.mark.skipif(not BIRD_DB_PATH.exists(), reason="data/financial.sqlite missing")
def test_build_reads_text_columns_from_the_database():
    index = ValueIndex.build(load_semantic_model().restricted_to(bird_tables()), BIRD_DB_PATH)
    assert index.entries["poplatek mesicne"] == ("POPLATEK MESICNE", ["account.frequency"])
    assert not any(c.startswith("district.A4") for _, cols in index.entries.values() for c in cols)
    assert len(index.entries) == 108
