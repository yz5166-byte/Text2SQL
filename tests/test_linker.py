from text2sql.semantic.linker import (
    find_ambiguous_terms,
    find_metric_mentions,
    find_value_mentions,
    link_value,
    link_value_any,
)
from text2sql.semantic.registry import DimValue, Dimension, load_semantic_model

model = load_semantic_model()


def keys(mentions):
    return [m.key for m in mentions]


def codes(dim, phrase):
    link = link_value(model, dim, phrase)
    return link.codes if link else None


def test_metric_longest_match():
    assert keys(find_metric_mentions("1998年10月南摩拉维亚的风险账户数有多少？", model)) == ["risk_account_cnt"]
    assert keys(find_metric_mentions("1997年不良贷款金额是多少？", model)) == ["bad_loan_amt"]  # not loan_amt
    assert keys(find_metric_mentions("1998年入账金额和出账金额分别是多少？", model)) == ["credit_amt", "debit_amt"]


def test_ambiguous_terms():
    assert keys(find_ambiguous_terms("上个月的损失有多少？", model)) == ["损失"]
    assert keys(find_ambiguous_terms("1997年的交易量是多少？", model)) == ["交易量"]
    assert find_ambiguous_terms("1998年10月的风险账户数", model) == []  # covered by a longer alias
    assert find_ambiguous_terms("1997年贷款损失", model) == []  # 贷款损失 is an alias of bad_loan_amt


def test_exact_links():
    assert codes("region", "南摩拉维亚") == ("south Moravia",)
    assert codes("region", "south moravia") == ("south Moravia",)
    assert codes("trans_operation", "取现") == ("VYBER",)
    assert codes("loan_status", "不良") == ("B", "D")
    assert codes("trans_direction", "出账") == ("VYDAJ", "VYBER")
    assert codes("trans_purpose", "未标注") == (None, " ")
    assert codes("district_name", "Brno mesto") == ("Brno - mesto",)  # equal after removing spaces and hyphens
    assert codes("loan_term", "5年期") == (60,)
    assert link_value(model, "region", "南摩拉维亚").fuzzy is False


def test_fuzzy_link_is_flagged_and_constrained():
    link = link_value(model, "district_name", "Plzen jh")  # typo of Plzen - jih
    assert link.codes == ("Plzen - jih",) and link.fuzzy
    assert link_value(model, "region", "南波西米亚") is None  # one wrong character: below 0.85, not guessed
    assert link_value(model, "district_name", "Praha") is None  # a region name, not linked to a district
    assert link_value(model, "region", "上海") is None


def test_tie_between_two_values_is_not_linked():
    dim = Dimension(key="t", name="测试", column="x.y", values=[
        DimValue(codes=["A"], label="abcdefgh1"), DimValue(codes=["B"], label="abcdefgh2")])
    tied = model.model_copy(update={"dimensions": {"t": dim}})
    assert link_value(tied, "t", "abcdefgh3") is None  # both score 0.89


def test_misplaced_filter_is_repairable():
    hits = link_value_any(model, ["trans_operation", "trans_purpose"], "透支罚息")
    assert [(d, link.codes) for d, link in hits] == [("trans_purpose", ("SANKC. UROK",))]


def test_find_value_mentions_in_text():
    found = find_value_mentions("1998年10月南摩拉维亚的风险账户数", model, ["region"])
    assert [(m.key, m.codes) for m in found] == [("region", ("south Moravia",))]
