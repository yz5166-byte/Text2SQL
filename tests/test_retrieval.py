"""Knowledge retrieval: tokenizer, BM25, RRF, the foreign-key bridge, and table recall on layer-3 questions."""

import pytest

from text2sql.eval.dataset import load_domain_eval
from text2sql.eval.retrieval import evaluate_retrieval, gold_tables
from text2sql.knowledge import embedding
from text2sql.knowledge.bm25 import BM25, tokenize
from text2sql.knowledge.retriever import TableRetriever, bridge_tables, foreign_key_graph, rrf
from text2sql.semantic.registry import load_semantic_model

model = load_semantic_model()
LAYER3 = [(i.question, i.gold_sql) for i in load_domain_eval() if i.route == "L3"]


def test_tokenize_mixes_english_words_and_chinese_bigrams():
    assert tokenize("account_id 交易流水表，卡") == ["account", "id", "交易", "易流", "流水", "水表", "卡"]


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
