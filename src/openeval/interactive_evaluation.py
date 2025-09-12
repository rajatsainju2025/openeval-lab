"""
Interactive Evaluation Mode for OpenEval Lab

This module provides real-time, interactive evaluation capabilities
with user feedback, progress tracking, and dynamic configuration.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from queue import Queue
import logging

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.text import Text

from .enhanced_logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationStep:
    """Represents a single evaluation step."""
    name: str
    description: str
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class InteractiveSession:
    """Manages an interactive evaluation session."""
    session_id: str
    config: Dict[str, Any]
    steps: List[EvaluationStep] = field(default_factory=list)
    current_step: int = 0
    status: str = "initialized"
    results: Dict[str, Any] = field(default_factory=dict)
    user_decisions: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, name: str, description: str) -> None:
        """Add an evaluation step."""
        step = EvaluationStep(name=name, description=description)
        self.steps.append(step)

    def update_step(self, step_index: int, **updates) -> None:
        """Update a step's status."""
        if 0 <= step_index < len(self.steps):
            step = self.steps[step_index]
            for key, value in updates.items():
                if hasattr(step, key):
                    setattr(step, key, value)

    def get_current_step(self) -> Optional[EvaluationStep]:
        """Get the currently executing step."""
        if 0 <= self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None


class InteractiveEvaluator:
    """
    Interactive evaluation system with real-time feedback and user control.
    """

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.sessions: Dict[str, InteractiveSession] = {}
        self.event_queue: Queue = Queue()
        self.running = False

    def start_interactive_session(
        self,
        config_path: Union[str, Path],
        session_id: Optional[str] = None
    ) -> str:
        """
        Start an interactive evaluation session.

        Args:
            config_path: Path to evaluation configuration
            session_id: Optional session identifier

        Returns:
            Session ID
        """
        if session_id is None:
            session_id = f"session_{int(time.time())}"

        # Load configuration
        config = self._load_config(config_path)

        # Create session
        session = InteractiveSession(
            session_id=session_id,
            config=config
        )

        # Initialize evaluation steps
        self._initialize_steps(session)

        self.sessions[session_id] = session

        logger.info(f"Started interactive session: {session_id}")
        return session_id

    def _load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load evaluation configuration."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix in ['.yaml', '.yml']:
                try:
                    import yaml
                    return yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML required for YAML configuration files")
            else:
                return json.load(f)

    def _initialize_steps(self, session: InteractiveSession) -> None:
        """Initialize evaluation steps based on configuration."""
        config = session.config

        # Basic evaluation pipeline
        session.add_step("config_validation", "Validating configuration")
        session.add_step("data_loading", "Loading evaluation datasets")
        session.add_step("model_initialization", "Initializing model adapters")
        session.add_step("baseline_evaluation", "Running baseline evaluation")
        session.add_step("metric_calculation", "Calculating evaluation metrics")
        session.add_step("result_analysis", "Analyzing results")
        session.add_step("report_generation", "Generating evaluation report")

        # Add task-specific steps
        task = config.get('task', 'qa')
        if task == 'code':
            session.add_step("syntax_check", "Checking code syntax")
            session.add_step("execution_test", "Testing code execution")
        elif task == 'summarization':
            session.add_step("readability_check", "Checking summary readability")
        elif task == 'generation':
            session.add_step("diversity_analysis", "Analyzing text diversity")

    async def run_interactive_evaluation(
        self,
        session_id: str,
        user_interface: bool = True
    ) -> Dict[str, Any]:
        """
        Run interactive evaluation with real-time feedback.

        Args:
            session_id: Session identifier
            user_interface: Whether to show interactive UI

        Returns:
            Evaluation results
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self.sessions[session_id]
        session.status = "running"

        try:
            if user_interface:
                return await self._run_with_ui(session)
            else:
                return await self._run_headless(session)
        finally:
            session.status = "completed"

    async def _run_with_ui(self, session: InteractiveSession) -> Dict[str, Any]:
        """Run evaluation with interactive user interface."""
        with Live(console=self.console, refresh_per_second=4) as live:
            # Initial display
            self._update_display(session, live)

            for i, step in enumerate(session.steps):
                session.current_step = i
                step.start_time = time.time()
                step.status = "running"

                self._update_display(session, live)

                try:
                    # Execute step
                    result = await self._execute_step(session, step)
                    step.result = result
                    step.status = "completed"
                    step.progress = 1.0

                except Exception as e:
                    step.status = "failed"
                    step.error = str(e)
                    logger.error(f"Step {step.name} failed: {e}")

                    # Ask user how to proceed
                    if await self._handle_error_interactively(session, step, live):
                        continue  # Retry
                    else:
                        break  # Stop evaluation

                step.end_time = time.time()
                self._update_display(session, live)

                # Allow user interaction between steps
                if not await self._step_transition_interactive(session, step, live):
                    break

        return self._compile_results(session)

    async def _run_headless(self, session: InteractiveSession) -> Dict[str, Any]:
        """Run evaluation without interactive UI."""
        for i, step in enumerate(session.steps):
            session.current_step = i
            step.start_time = time.time()
            step.status = "running"

            try:
                result = await self._execute_step(session, step)
                step.result = result
                step.status = "completed"
                step.progress = 1.0
            except Exception as e:
                step.status = "failed"
                step.error = str(e)
                logger.error(f"Step {step.name} failed: {e}")
                break

            step.end_time = time.time()

        return self._compile_results(session)

    async def _execute_step(
        self,
        session: InteractiveSession,
        step: EvaluationStep
    ) -> Any:
        """Execute a single evaluation step."""
        # Simulate step execution with progress updates
        import random

        # Mock execution - in real implementation, this would call actual evaluation logic
        for progress in [0.1, 0.3, 0.6, 0.8, 1.0]:
            step.progress = progress
            await asyncio.sleep(0.5 + random.random() * 0.5)

        # Mock results based on step name
        if step.name == "config_validation":
            return {"valid": True, "warnings": []}
        elif step.name == "data_loading":
            return {"datasets_loaded": 3, "total_samples": 1500}
        elif step.name == "model_initialization":
            return {"models_ready": 2, "adapters": ["openai", "local"]}
        elif step.name == "baseline_evaluation":
            return {"accuracy": 0.85, "f1_score": 0.82}
        elif step.name == "metric_calculation":
            return {
                "accuracy": 0.85,
                "precision": 0.87,
                "recall": 0.83,
                "f1": 0.82,
                "bleu": 0.75
            }
        else:
            return {"status": "completed"}

    def _update_display(self, session: InteractiveSession, live: Live) -> None:
        """Update the interactive display."""
        table = Table(title=f"🔬 Interactive Evaluation - {session.session_id}")
        table.add_column("Step", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Progress", style="yellow")
        table.add_column("Time", style="blue")

        for step in session.steps:
            status_icon = {
                "pending": "⏳",
                "running": "🔄",
                "completed": "✅",
                "failed": "❌"
            }.get(step.status, "?")

            progress_bar = f"[{'█' * int(step.progress * 10)}{'░' * (10 - int(step.progress * 10))}]"

            duration = ""
            if step.start_time and step.end_time:
                duration = ".1f"
            elif step.start_time:
                duration = ".1f"

            table.add_row(
                step.name,
                f"{status_icon} {step.status}",
                progress_bar,
                duration
            )

        # Add current step details
        current_step = session.get_current_step()
        if current_step:
            details = f"**Current:** {current_step.description}"
            if current_step.error:
                details += f"\n**Error:** {current_step.error}"
        else:
            details = "Evaluation completed"

        panel = Panel.fit(
            f"{table}\n\n{details}",
            title="📊 Evaluation Progress",
            border_style="blue"
        )

        live.update(panel)

    async def _handle_error_interactively(
        self,
        session: InteractiveSession,
        step: EvaluationStep,
        live: Live
    ) -> bool:
        """Handle errors interactively."""
        # In a real implementation, this would prompt the user
        # For now, just log and continue
        logger.warning(f"Step {step.name} failed: {step.error}")
        return False  # Don't retry

    async def _step_transition_interactive(
        self,
        session: InteractiveSession,
        step: EvaluationStep,
        live: Live
    ) -> bool:
        """Handle transitions between steps interactively."""
        # In a real implementation, this would allow user to modify config
        # or skip steps
        return True  # Continue

    def _compile_results(self, session: InteractiveSession) -> Dict[str, Any]:
        """Compile final evaluation results."""
        results = {
            "session_id": session.session_id,
            "status": session.status,
            "total_steps": len(session.steps),
            "completed_steps": sum(1 for s in session.steps if s.status == "completed"),
            "failed_steps": sum(1 for s in session.steps if s.status == "failed"),
            "step_results": {}
        }

        total_time = 0.0
        for step in session.steps:
            if step.start_time and step.end_time:
                step_time = step.end_time - step.start_time
                total_time += step_time
                results["step_results"][step.name] = {
                    "status": step.status,
                    "duration": step_time,
                    "result": step.result,
                    "error": step.error
                }

        results["total_duration"] = total_time
        return results

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an evaluation session."""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]
        return {
            "session_id": session.session_id,
            "status": session.status,
            "current_step": session.current_step,
            "total_steps": len(session.steps),
            "progress": session.current_step / len(session.steps) if session.steps else 0
        }

    def pause_session(self, session_id: str) -> bool:
        """Pause an evaluation session."""
        if session_id in self.sessions:
            self.sessions[session_id].status = "paused"
            return True
        return False

    def resume_session(self, session_id: str) -> bool:
        """Resume a paused evaluation session."""
        if session_id in self.sessions:
            self.sessions[session_id].status = "running"
            return True
        return False

    def stop_session(self, session_id: str) -> bool:
        """Stop an evaluation session."""
        if session_id in self.sessions:
            self.sessions[session_id].status = "stopped"
            return True
        return False


def create_interactive_evaluator() -> InteractiveEvaluator:
    """Create an interactive evaluator instance."""
    return InteractiveEvaluator()


async def run_interactive_evaluation(
    config_path: Union[str, Path],
    session_id: Optional[str] = None,
    headless: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to run interactive evaluation.

    Args:
        config_path: Path to evaluation configuration
        session_id: Optional session identifier
        headless: Run without interactive UI

    Returns:
        Evaluation results
    """
    evaluator = InteractiveEvaluator()
    session_id = evaluator.start_interactive_session(config_path, session_id)
    return await evaluator.run_interactive_evaluation(session_id, user_interface=not headless)