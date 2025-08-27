"""
Test trace collection for ToolUseTask with --traces flag.
"""
import json
from pathlib import Path
import tempfile

from openeval.tasks.tooluse import ToolUseTask
from openeval.adapters.echo import EchoAdapter
from openeval.datasets.inline import InlineDataset
from openeval.metrics.tool_execution import ToolExecutionMetric


def test_tooluse_task_with_traces():
    """Test that ToolUseTask emits traces when _collect_traces is True."""
    adapter = EchoAdapter()
    task = ToolUseTask(
        agent_type="openeval.agents.react.ReActAgent", 
        tools=["openeval.tools.calculator.Calculator"]
    )
    # Simulate CLI setting this attribute
    setattr(task, "_collect_traces", True)
    
    ds = InlineDataset(name="toy", examples=[
        {"id": "1", "input": "2+3", "reference": "5"},
    ])
    
    out = task.evaluate(adapter, ds, [ToolExecutionMetric()], seed=0, collect_records=True)
    
    assert "records" in out
    assert len(out["records"]) == 1
    
    record = out["records"][0]
    assert "trace" in record
    assert isinstance(record["trace"], list)
    # Since EchoAdapter returns input as output, ReActAgent will have some fallback behavior
    # but we should still see a trace structure
    if record["trace"]:
        assert all("thought" in step and "action" in step for step in record["trace"])


def test_tooluse_task_without_traces():
    """Test that ToolUseTask doesn't emit traces when _collect_traces is False."""
    adapter = EchoAdapter()
    task = ToolUseTask(
        agent_type="openeval.agents.react.ReActAgent", 
        tools=["openeval.tools.calculator.Calculator"]
    )
    # Default behavior: no traces
    
    ds = InlineDataset(name="toy", examples=[
        {"id": "1", "input": "2+3", "reference": "5"},
    ])
    
    out = task.evaluate(adapter, ds, [ToolExecutionMetric()], seed=0, collect_records=True)
    
    assert "records" in out
    assert len(out["records"]) == 1
    
    record = out["records"][0]
    assert "trace" not in record
