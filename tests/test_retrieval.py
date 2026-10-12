"""Knowledge retrieval: tokenizer, BM25, RRF, the foreign-key bridge, and table recall on layer-3 questions."""

import pytest

from text2sql.eval.dataset import load_domain_eval
from text2sql.eval.retrieval import evaluate_retrieval, gold_columns, gold_tables, gold_values
from text2sql.knowledge import embedding
from text2sql.knowledge.bm25 import BM25, tokenize
from text2sql.knowledge.retriever import ColumnRetriever, TableRetriever, bridge_tables, foreign_key_graph, rrf
from text2sql.knowledge.units import column_units
from text2sql.semantic.registry import load_semantic_model

model = load_semantic_model()
LAYER3 = [(i.question, i.gold_sql) for i in load_domain_eval() if i.route == "L3"]


def test_tokenize_mixes_english_words_and_chinese_bigrams():
    assert tokenize("account_id 交易流水表，卡") == ["account", "id", "交易", "易流", "流水", "水表", "卡"]


def test_english_words_lose_their_plural_ending():
    assert tokenize("Accounts salaries class loans_id 账户") == ["account", "salary", "class", "loan", "id", "账户"]


def test_a_query_without_chinese_is_ranked_by_bm25_alone():
    from text2sql.knowledge.retriever import HybridIndex
    index = HybridIndex({"loan": "loan amount 贷款", "card": "card type 卡"}, "hybrid", bm25_unless_chinese=True)
    assert index.rank("loan amounts")[0] == "loan" and index._vectors is None  # no embedding computed
    # only the schema retrievers, and only with a Chinese-only model; examples always keep the fusion
    assert TableRetriever(model).index.bm25_unless_chinese == (not embedding.multilingual())


def test_bm25_prefers_the_rarer_shared_term():
    scores = BM25({"a": "交易 对手 账号", "b": "交易 日期", "c": "交易 金额"}).scores("交易对手")
    assert max(scores, key=scores.get) == "a"


def test_rrf_rewards_agreement_between_rankings():
    fused = rrf({"x": 9, "y": 5, "z": 1}, {"x": 0.2, "y": 0.9, "z": 0.1})
    assert sorted(fused, key=lambda k: -fused[k])[-1] == "z"


def test_foreign_key_graph_and_bridge():
    graph = foreign_key_graph(model)
    assert {"client", "account", "card"} <= graph["disp"] and graph["ads_okr_monthly"] == set()
    bridged = bridge_tables(["client", "loan"], graph)  # client - disp - account - loan
    assert {"disp", "account"} <= set(bridged) and not {"client", "loan"} & set(bridged)
    assert bridge_tables(["loan", "ads_okr_monthly"], graph) == []  # unconnected: nothing to add


def test_gold_tables_ignore_cte_names():
    sql = "WITH big AS (SELECT * FROM LOAN l JOIN Account a USING (account_id)) SELECT * FROM big"
    assert gold_tables(sql, model.tables) == {"loan", "account"}


def test_bm25_table_recall_on_layer3_questions():
    report = evaluate_retrieval(TableRetriever(model, "bm25", k=5), LAYER3)
    assert report.table_recall >= 0.95 and report.complete >= 29  # measured: 98.4%, 29/30


@pytest.mark.skipif(not embedding.model_cached(), reason="embedding model not downloaded")
def test_hybrid_table_recall_on_layer3_questions():
    report = evaluate_retrieval(TableRetriever(model, "hybrid", k=5), LAYER3)
    assert report.table_recall >= 0.95 and report.complete >= 29  # measured: 98.4%, 29/30


def test_gold_columns_resolve_aliases_subqueries_and_ctes():
    sql = """WITH big AS (SELECT l.account_id, l.amount FROM loan l WHERE l.status IN ('B', 'D'))
             SELECT d.A3, SUM(big.amount) FROM big JOIN account a ON a.account_id = big.account_id
             JOIN district d ON d.district_id = a.district_id
             WHERE a.district_id IN (SELECT district_id FROM district WHERE A2 = 'Most') GROUP BY d.A3"""
    assert gold_columns(sql, model) == {"loan.account_id", "loan.amount", "loan.status", "account.account_id",
                                        "account.district_id", "district.district_id", "district.A3", "district.A2"}
    assert gold_values(sql, model) == {("loan.status", "B"), ("loan.status", "D"), ("district.A2", "Most")}


def test_detailed_column_documents_add_descriptions_and_codes():
    short, detailed = column_units(model)["card.type"], column_units(model, detailed=True)["card.type"]
    assert short.startswith("card.type 信用卡表 卡等级") and "金卡" not in short
    assert detailed.startswith(short) and "金卡" in detailed


def test_column_ranking_within_tables():
    retriever = ColumnRetriever(model, "bm25")
    assert retriever.rank("卡等级")[0] == "card.type"
    assert {c.split(".")[0] for c in retriever.rank("卡等级", ["loan", "account"])} == {"loan", "account"}


def test_linked_tables_join_the_candidates_before_the_bridge():
    retriever = TableRetriever(model, "bm25", k=1)
    plain = retriever.retrieve("放款金额")
    linked = retriever.retrieve("放款金额", ["district", "loan"])
    assert plain.tables == linked.tables == ["loan"] and plain.linked == [] and plain.bridged == []
    assert linked.linked == ["district"] and linked.bridged == ["account"]  # loan - account - district
    assert linked.all == ["loan", "district", "account"]


@pytest.mark.skipif(not embedding.model_cached(), reason="embedding model not downloaded")
def test_vectors_are_cached_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(embedding, "VECTOR_DIR", tmp_path)
    first = embedding.embed(["loan amount", "贷款金额"])
    assert len(list(tmp_path.rglob("*.npy"))) == 2
    assert abs(float(embedding.embed(["loan amount"])[0] @ first[0]) - 1) < 1e-5


@pytest.mark.skipif(not (embedding.model_cached() and embedding.multilingual()), reason="needs the multilingual model")
def test_the_multilingual_model_puts_chinese_and_english_together():
    zh, en, other = embedding.embed(["每个地区的贷款金额", "loan amount by region", "card type of the client"])
    assert zh @ en > zh @ other + 0.2
