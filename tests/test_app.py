"""Streamlit demo, driven headless with AppTest and without a model (rule paths only)."""

import importlib.util
import sys
from pathlib import Path

import pytest

from text2sql.config import RISK_DB_PATH

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = Path(__file__).parents[1] / "app" / "streamlit_app.py"
needs_db = pytest.mark.skipif(not RISK_DB_PATH.exists(), reason="data/bank_risk.sqlite missing")


def start() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    return at.selectbox[0].set_value("无模型（仅规则）").run()


def shown(at: AppTest) -> str:
    """Every markdown, caption, warning and error on the page, markdown escapes removed."""
    parts = [e.value for kind in (at.markdown, at.caption, at.warning, at.error) for e in kind]
    return "\n".join(parts).replace("\\", "")


def submit_time(at: AppTest, turn: int, reply: str) -> AppTest:
    at.text_input(key=f"time-input-{turn}").input(reply)
    return next(b for b in at.button if b.label == "提交").click().run()


@needs_db
def test_an_example_is_answered_with_title_table_and_details():
    at = start()
    at.button(key="example-L1-01").click().run()
    assert "第一层 OKR 物化表 · 快速通道" in shown(at) and "风险事件数 265起" in shown(at)
    assert at.dataframe[0].value.iloc[0, 0] == "265"
    assert "ads_okr_monthly" in at.code[0].value and "模型调用 0 次" in shown(at)


@needs_db
def test_a_metric_then_a_time_clarification_are_answered_in_place():
    at = start()
    at.chat_input[0].set_value("最近的交易量是多少？").run()
    at.button(key="metric-0-0").click().run()
    assert "重新提问：最近的交易笔数是多少？" in shown(at)
    at = submit_time(at, 1, "下下个世纪")
    assert "没有识别出时间" in shown(at) and len(at.session_state.turns) == 2
    at = submit_time(at, 1, "1998年11月")
    assert "重新提问：1998年11月的交易笔数是多少？" in shown(at) and "交易笔数 25,341笔" in shown(at)


@needs_db
def test_clarification_stops_after_three_rounds():
    at = start()
    at.chat_input[0].set_value("1999年的风险事件数是多少？").run()
    for turn, year in enumerate(["2000年", "2001年", "2002年"]):
        at = submit_time(at, turn, year)
    assert "已追问 3 轮" in shown(at) and not [b for b in at.button if b.label == "提交"]


def test_a_routing_failure_is_shown_as_unfinished():
    at = start()
    at.chat_input[0].set_value("平均工资最高的区县是哪个？").run()
    assert "未完成 · 路由失败" in at.error[0].value


def test_text_is_escaped_and_columns_are_unique(monkeypatch):
    spec = importlib.util.spec_from_file_location("streamlit_app", APP)
    app = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, app)  # dataclasses look their module up while it loads
    spec.loader.exec_module(app)  # main() only runs under streamlit
    assert app.plain("COUNT(*) a_b $5") == r"COUNT\(\*\) a\_b \$5"
    assert app.unique_columns(["区县", "金额", "区县"]) == ["区县", "金额", "区县（2）"]
