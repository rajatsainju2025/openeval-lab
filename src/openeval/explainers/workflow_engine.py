"""Workflow engine for defining explanation generation workflows.

This module provides a workflow engine for orchestrating multi-step
explanation generation processes with conditional logic and branching.

Example:
    >>> from openeval.explainers import WorkflowEngine, create_workflow
    >>> engine = WorkflowEngine()
    >>> workflow = engine.create_workflow("analysis")
    >>> workflow.add_step("parse", parse_code)
    >>> workflow.add_step("explain", explain_code)
    >>> result = await engine.execute(workflow, code="def foo(): pass")
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, TypeVar


class StepStatus(Enum):
    """Status of a workflow step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class WorkflowStatus(Enum):
    """Status of a workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepType(Enum):
    """Types of workflow steps."""

    TASK = "task"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    LOOP = "loop"
    SUBWORKFLOW = "subworkflow"


T = TypeVar("T")


@dataclass
class StepResult:
    """Result of a workflow step execution."""

    step_name: str
    status: StepStatus
    output: Any = None
    error: str | None = None
    duration: float = 0.0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""

    workflow_name: str
    status: WorkflowStatus
    steps: list[StepResult]
    output: Any = None
    error: str | None = None
    total_duration: float = 0.0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowContext:
    """Context passed through workflow steps."""

    workflow_name: str
    data: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from context."""
        return self.data.get(key, self.variables.get(key, default))

    def set(self, key: str, value: Any) -> None:
        """Set a value in context."""
        self.data[key] = value

    def set_result(self, step_name: str, value: Any) -> None:
        """Set result from a step."""
        self.results[step_name] = value


class WorkflowStep(ABC):
    """Abstract base class for workflow steps."""

    def __init__(
        self,
        name: str,
        step_type: StepType = StepType.TASK,
    ) -> None:
        """Initialize the step."""
        self.name = name
        self.step_type = step_type
        self.condition: Callable[[WorkflowContext], bool] | None = None
        self.on_error: str | None = None
        self.timeout: float | None = None
        self.retry_count: int = 0
        self.metadata: dict[str, Any] = {}

    @abstractmethod
    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the step."""
        pass

    def should_run(self, context: WorkflowContext) -> bool:
        """Check if step should run based on condition."""
        if self.condition is None:
            return True
        return self.condition(context)


class TaskStep(WorkflowStep):
    """A simple task step."""

    def __init__(
        self,
        name: str,
        func: Callable[..., Any],
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the task step."""
        super().__init__(name, StepType.TASK)
        self.func = func
        self.args = args or ()
        self.kwargs = kwargs or {}

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the task."""
        # Merge kwargs with context data
        merged_kwargs = {**self.kwargs}
        for key in merged_kwargs:
            if merged_kwargs[key] == "__context__":
                merged_kwargs[key] = context

        # Check if function is async
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(*self.args, **merged_kwargs)
        return self.func(*self.args, **merged_kwargs)


class ConditionalStep(WorkflowStep):
    """A conditional branching step."""

    def __init__(
        self,
        name: str,
        condition: Callable[[WorkflowContext], bool],
        if_true: WorkflowStep,
        if_false: WorkflowStep | None = None,
    ) -> None:
        """Initialize the conditional step."""
        super().__init__(name, StepType.CONDITIONAL)
        self._condition = condition
        self.if_true = if_true
        self.if_false = if_false

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the conditional step."""
        if self._condition(context):
            return await self.if_true.execute(context)
        elif self.if_false:
            return await self.if_false.execute(context)
        return None


class ParallelStep(WorkflowStep):
    """A step that executes multiple steps in parallel."""

    def __init__(
        self,
        name: str,
        steps: list[WorkflowStep],
        fail_fast: bool = True,
    ) -> None:
        """Initialize the parallel step."""
        super().__init__(name, StepType.PARALLEL)
        self.steps = steps
        self.fail_fast = fail_fast

    async def execute(self, context: WorkflowContext) -> list[Any]:
        """Execute steps in parallel."""
        tasks = [step.execute(context) for step in self.steps]

        if self.fail_fast:
            return await asyncio.gather(*tasks)
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results


class LoopStep(WorkflowStep):
    """A step that loops over items."""

    def __init__(
        self,
        name: str,
        items_key: str,
        step: WorkflowStep,
        item_key: str = "current_item",
    ) -> None:
        """Initialize the loop step."""
        super().__init__(name, StepType.LOOP)
        self.items_key = items_key
        self.step = step
        self.item_key = item_key

    async def execute(self, context: WorkflowContext) -> list[Any]:
        """Execute step for each item."""
        items = context.get(self.items_key, [])
        results = []

        for item in items:
            context.set(self.item_key, item)
            result = await self.step.execute(context)
            results.append(result)

        return results


@dataclass
class WorkflowDefinition:
    """Definition of a workflow."""

    name: str
    description: str = ""
    steps: list[WorkflowStep] = field(default_factory=list)
    version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(
        self,
        name: str,
        func: Callable[..., Any],
        **kwargs: Any,
    ) -> WorkflowDefinition:
        """Add a task step to the workflow."""
        step = TaskStep(name, func, kwargs=kwargs)
        self.steps.append(step)
        return self

    def add_step_instance(self, step: WorkflowStep) -> WorkflowDefinition:
        """Add a step instance to the workflow."""
        self.steps.append(step)
        return self

    def add_conditional(
        self,
        name: str,
        condition: Callable[[WorkflowContext], bool],
        if_true: WorkflowStep,
        if_false: WorkflowStep | None = None,
    ) -> WorkflowDefinition:
        """Add a conditional step."""
        step = ConditionalStep(name, condition, if_true, if_false)
        self.steps.append(step)
        return self

    def add_parallel(
        self,
        name: str,
        steps: list[WorkflowStep],
        fail_fast: bool = True,
    ) -> WorkflowDefinition:
        """Add a parallel execution step."""
        step = ParallelStep(name, steps, fail_fast)
        self.steps.append(step)
        return self

    def add_loop(
        self,
        name: str,
        items_key: str,
        step: WorkflowStep,
        item_key: str = "current_item",
    ) -> WorkflowDefinition:
        """Add a loop step."""
        loop_step = LoopStep(name, items_key, step, item_key)
        self.steps.append(loop_step)
        return self


class WorkflowExecutor:
    """Executes workflows."""

    def __init__(self) -> None:
        """Initialize the executor."""
        self._running_workflows: dict[str, WorkflowContext] = {}
        self._listeners: dict[str, list[Callable[[StepResult], None]]] = {}

    async def execute(
        self,
        workflow: WorkflowDefinition,
        context: WorkflowContext | None = None,
        **initial_data: Any,
    ) -> WorkflowResult:
        """Execute a workflow.

        Args:
            workflow: Workflow definition.
            context: Optional initial context.
            **initial_data: Initial data for context.

        Returns:
            WorkflowResult with execution details.
        """
        import time

        start_time = time.time()
        started_at = datetime.now()

        # Initialize context
        if context is None:
            context = WorkflowContext(
                workflow_name=workflow.name,
                data=initial_data,
            )
        else:
            context.data.update(initial_data)

        self._running_workflows[workflow.name] = context

        step_results: list[StepResult] = []
        workflow_status = WorkflowStatus.RUNNING
        final_output: Any = None
        error_message: str | None = None

        try:
            for step in workflow.steps:
                step_result = await self._execute_step(step, context)
                step_results.append(step_result)

                # Notify listeners
                self._notify_listeners(workflow.name, step_result)

                if step_result.status == StepStatus.FAILED:
                    if step.on_error:
                        # Handle error step
                        continue
                    workflow_status = WorkflowStatus.FAILED
                    error_message = step_result.error
                    break

                # Store step result in context
                if step_result.output is not None:
                    context.set_result(step.name, step_result.output)
                    final_output = step_result.output

            if workflow_status == WorkflowStatus.RUNNING:
                workflow_status = WorkflowStatus.COMPLETED

        except asyncio.CancelledError:
            workflow_status = WorkflowStatus.CANCELLED
        except Exception as exc:
            workflow_status = WorkflowStatus.FAILED
            error_message = str(exc)
        finally:
            del self._running_workflows[workflow.name]

        return WorkflowResult(
            workflow_name=workflow.name,
            status=workflow_status,
            steps=step_results,
            output=final_output,
            error=error_message,
            total_duration=time.time() - start_time,
            started_at=started_at,
            completed_at=datetime.now(),
        )

    async def _execute_step(self, step: WorkflowStep, context: WorkflowContext) -> StepResult:
        """Execute a single step."""
        import time

        start_time = time.time()
        started_at = datetime.now()

        # Check condition
        if not step.should_run(context):
            return StepResult(
                step_name=step.name,
                status=StepStatus.SKIPPED,
                started_at=started_at,
                completed_at=datetime.now(),
            )

        status = StepStatus.RUNNING
        output: Any = None
        error: str | None = None

        try:
            # Execute with timeout if specified
            if step.timeout:
                output = await asyncio.wait_for(
                    step.execute(context),
                    timeout=step.timeout,
                )
            else:
                output = await step.execute(context)

            status = StepStatus.COMPLETED

        except asyncio.TimeoutError:
            status = StepStatus.FAILED
            error = f"Step timed out after {step.timeout} seconds"
        except Exception as exc:
            status = StepStatus.FAILED
            error = str(exc)

            # Retry if configured
            if step.retry_count > 0:
                for attempt in range(step.retry_count):
                    try:
                        output = await step.execute(context)
                        status = StepStatus.COMPLETED
                        error = None
                        break
                    except Exception as retry_exc:
                        error = str(retry_exc)

        return StepResult(
            step_name=step.name,
            status=status,
            output=output,
            error=error,
            duration=time.time() - start_time,
            started_at=started_at,
            completed_at=datetime.now(),
        )

    def add_listener(
        self,
        workflow_name: str,
        listener: Callable[[StepResult], None],
    ) -> None:
        """Add a step result listener."""
        if workflow_name not in self._listeners:
            self._listeners[workflow_name] = []
        self._listeners[workflow_name].append(listener)

    def _notify_listeners(self, workflow_name: str, result: StepResult) -> None:
        """Notify listeners of step completion."""
        for listener in self._listeners.get(workflow_name, []):
            try:
                listener(result)
            except Exception:
                pass  # Don't let listener errors affect workflow


class WorkflowEngine:
    """Main workflow engine class."""

    def __init__(self) -> None:
        """Initialize the workflow engine."""
        self._workflows: dict[str, WorkflowDefinition] = {}
        self._executor = WorkflowExecutor()
        self._templates: dict[str, WorkflowDefinition] = {}

    def create_workflow(
        self,
        name: str,
        description: str = "",
    ) -> WorkflowDefinition:
        """Create a new workflow.

        Args:
            name: Workflow name.
            description: Workflow description.

        Returns:
            New WorkflowDefinition.
        """
        workflow = WorkflowDefinition(name=name, description=description)
        self._workflows[name] = workflow
        return workflow

    def register_workflow(self, workflow: WorkflowDefinition) -> None:
        """Register a workflow definition."""
        self._workflows[workflow.name] = workflow

    def get_workflow(self, name: str) -> WorkflowDefinition | None:
        """Get a workflow by name."""
        return self._workflows.get(name)

    def list_workflows(self) -> list[str]:
        """List all registered workflow names."""
        return list(self._workflows.keys())

    async def execute(
        self,
        workflow: WorkflowDefinition | str,
        **initial_data: Any,
    ) -> WorkflowResult:
        """Execute a workflow.

        Args:
            workflow: Workflow definition or name.
            **initial_data: Initial data for context.

        Returns:
            WorkflowResult with execution details.
        """
        if isinstance(workflow, str):
            workflow_def = self._workflows.get(workflow)
            if not workflow_def:
                raise ValueError(f"Workflow '{workflow}' not found")
        else:
            workflow_def = workflow

        return await self._executor.execute(workflow_def, **initial_data)

    def add_template(self, name: str, workflow: WorkflowDefinition) -> None:
        """Add a workflow template."""
        self._templates[name] = workflow

    def from_template(
        self,
        template_name: str,
        new_name: str,
    ) -> WorkflowDefinition | None:
        """Create a workflow from a template."""
        template = self._templates.get(template_name)
        if not template:
            return None

        # Create a copy with new name
        workflow = WorkflowDefinition(
            name=new_name,
            description=template.description,
            steps=list(template.steps),
            version=template.version,
            metadata=dict(template.metadata),
        )

        self._workflows[new_name] = workflow
        return workflow

    def add_listener(
        self,
        workflow_name: str,
        listener: Callable[[StepResult], None],
    ) -> None:
        """Add a step result listener."""
        self._executor.add_listener(workflow_name, listener)

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "registered_workflows": len(self._workflows),
            "templates": len(self._templates),
            "workflow_names": list(self._workflows.keys()),
        }


# Pre-built workflow templates
def create_explanation_workflow(name: str = "explanation") -> WorkflowDefinition:
    """Create a standard explanation workflow template.

    Returns:
        WorkflowDefinition for code explanation.
    """
    workflow = WorkflowDefinition(
        name=name,
        description="Standard code explanation workflow",
    )

    # Placeholder steps - users should customize
    async def parse_code(context: WorkflowContext) -> dict[str, Any]:
        code = context.get("code", "")
        return {"code": code, "parsed": True}

    async def analyze_code(context: WorkflowContext) -> dict[str, Any]:
        return {"analysis": "Code analyzed"}

    async def generate_explanation(context: WorkflowContext) -> str:
        return "Generated explanation"

    workflow.add_step("parse", parse_code)
    workflow.add_step("analyze", analyze_code)
    workflow.add_step("explain", generate_explanation)

    return workflow


def create_batch_workflow(name: str = "batch") -> WorkflowDefinition:
    """Create a batch processing workflow template.

    Returns:
        WorkflowDefinition for batch processing.
    """
    workflow = WorkflowDefinition(
        name=name,
        description="Batch processing workflow",
    )

    async def process_item(context: WorkflowContext) -> Any:
        item = context.get("current_item")
        return {"item": item, "processed": True}

    process_step = TaskStep("process_item", process_item)

    workflow.add_loop("batch_process", "items", process_step)

    return workflow


# Global instance
_workflow_engine: WorkflowEngine | None = None


def get_workflow_engine() -> WorkflowEngine:
    """Get the global workflow engine."""
    global _workflow_engine
    if _workflow_engine is None:
        _workflow_engine = WorkflowEngine()
    return _workflow_engine


def reset_workflow_engine() -> None:
    """Reset the global workflow engine."""
    global _workflow_engine
    _workflow_engine = None


def create_workflow_engine() -> WorkflowEngine:
    """Create a new workflow engine.

    Returns:
        New WorkflowEngine instance.
    """
    return WorkflowEngine()


def create_workflow(name: str, description: str = "") -> WorkflowDefinition:
    """Create a new workflow.

    Args:
        name: Workflow name.
        description: Workflow description.

    Returns:
        New WorkflowDefinition.
    """
    return get_workflow_engine().create_workflow(name, description)


async def run_workflow(workflow: WorkflowDefinition | str, **data: Any) -> WorkflowResult:
    """Run a workflow.

    Args:
        workflow: Workflow definition or name.
        **data: Initial data.

    Returns:
        WorkflowResult.
    """
    return await get_workflow_engine().execute(workflow, **data)


def create_task_step(
    name: str,
    func: Callable[..., Any],
    **kwargs: Any,
) -> TaskStep:
    """Create a task step.

    Args:
        name: Step name.
        func: Step function.
        **kwargs: Function arguments.

    Returns:
        TaskStep instance.
    """
    return TaskStep(name, func, kwargs=kwargs)


def create_parallel_step(
    name: str,
    steps: list[WorkflowStep],
    fail_fast: bool = True,
) -> ParallelStep:
    """Create a parallel step.

    Args:
        name: Step name.
        steps: Steps to run in parallel.
        fail_fast: Whether to fail on first error.

    Returns:
        ParallelStep instance.
    """
    return ParallelStep(name, steps, fail_fast)


def create_loop_step(
    name: str,
    items_key: str,
    step: WorkflowStep,
) -> LoopStep:
    """Create a loop step.

    Args:
        name: Step name.
        items_key: Context key for items.
        step: Step to run for each item.

    Returns:
        LoopStep instance.
    """
    return LoopStep(name, items_key, step)
