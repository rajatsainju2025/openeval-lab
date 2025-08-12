from pathlib import Path

from openeval.metrics.judge import LLMJudge
from openeval.adapters.echo import EchoAdapter


def test_llm_judge_echo_ties():
    # With echo adapter, the judge returns the prompt; heuristic maps to Tie most times
    m = LLMJudge(judge_adapter="echo")
    res = m.compute(["pred"], ["ref"])  # no prompt context
    assert set(res.keys()) == {"win_rate", "wins", "losses", "ties"}
    assert res["wins"] + res["losses"] + res["ties"] == 1


def test_llm_judge_order_balance():
    # Force deterministic outcomes by patching adapter.generate
    class FakeAdapter(EchoAdapter):
        def __init__(self):
            super().__init__()
            self.calls = []

        def generate(self, prompt: str, **kwargs):
            # Return 'A' on first call, 'B' on second call
            self.calls.append(prompt)
            return "A" if len(self.calls) % 2 == 1 else "B"

    m = LLMJudge(judge_adapter="echo")
    # monkeypatch load to use FakeAdapter
    m._load_adapter = lambda: FakeAdapter()
    res = m.compute(["p1"], ["r1"])
    # A then B => tie
    assert res["ties"] == 1
