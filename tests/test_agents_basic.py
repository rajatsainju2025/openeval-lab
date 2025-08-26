from __future__ import annotations

import json
from pathlib import Path

from openeval.tasks.tooluse import ToolUseTask
from openeval.adapters.echo import EchoAdapter
from openeval.agents.react import ReActAgent
from openeval.tools.calculator import Calculator
from openeval.datasets.inline import InlineDataset
from openeval.metrics.tool_execution import ToolExecutionMetric


def test_tool_execution_metric_with_strings():
    m = ToolExecutionMetric()
    res = m.compute(["5", "14.0"], ["5", "14.0"])
    assert res["accuracy"] == 1.0


def test_tool_use_task_with_agent_loop():
    # Assemble agentic components directly
    adapter = EchoAdapter()
    agent = ReActAgent(adapter, [Calculator()])
    task = ToolUseTask(agent_type="openeval.agents.react.ReActAgent", tools=["openeval.tools.calculator.Calculator"])  # agent created inside
    ds = InlineDataset(name="toy", examples=[
        {"id": "1", "input": "2+3", "reference": "5"},
        {"id": "2", "input": "10-7", "reference": "3"}
    ])

    # Because EchoAdapter doesn't actually plan, ReActAgent will fallback to direct answer; predictions will be arbitrary.
    # This test ensures evaluate executes and returns a payload with expected fields.
    out = task.evaluate(adapter, ds, [ToolExecutionMetric()], seed=0, collect_records=True)
    assert out["task"] == "tool_use"
    assert out["dataset"] == "toy"
    assert "metrics" in out
    assert "records" in out
