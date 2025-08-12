"""Tests for the curated task library."""

import pytest
import tempfile
import os
import json

from openeval.library import (
    TaskLibrary,
    BenchmarkSuite,
    get_task_library,
    list_available_tasks,
    get_task_info,
)


class TestTaskLibrary:
    """Test the task library functionality."""

    def test_library_initialization(self):
        """Test that library initializes with default tasks."""
        lib = TaskLibrary()

        # Should have some default tasks
        tasks = lib.list_tasks()
        assert len(tasks) > 0

        # Should have multiple categories
        categories = lib.list_categories()
        assert len(categories) > 0
        assert "question_answering" in categories
        assert "multiple_choice" in categories
        assert "code_generation" in categories

    def test_get_task(self):
        """Test getting individual tasks."""
        lib = TaskLibrary()

        # Test getting existing task
        task = lib.get_task("qa_basic")
        assert task is not None
        assert task["id"] == "qa_basic"
        assert task["category"] == "question_answering"
        assert "spec" in task

        # Test getting non-existent task
        task = lib.get_task("nonexistent")
        assert task is None

    def test_get_task_spec(self):
        """Test getting task specs as EvalSpec objects."""
        lib = TaskLibrary()

        spec = lib.get_task_spec("qa_basic")
        assert spec is not None
        # Should be a valid spec with required fields
        assert hasattr(spec, "task")
        assert hasattr(spec, "dataset")
        assert hasattr(spec, "adapter")
        assert hasattr(spec, "metrics")

    def test_list_tasks_filtering(self):
        """Test task listing with filters."""
        lib = TaskLibrary()

        # Test category filtering
        qa_tasks = lib.list_tasks(category="question_answering")
        assert len(qa_tasks) > 0
        assert all(t["category"] == "question_answering" for t in qa_tasks)

        # Test listing all tasks
        all_tasks = lib.list_tasks()
        assert len(all_tasks) >= len(qa_tasks)

    def test_search_tasks(self):
        """Test task search functionality."""
        lib = TaskLibrary()

        # Search by partial ID
        results = lib.search_tasks("qa")
        assert len(results) > 0
        assert any("qa" in t["id"] for t in results)

        # Search by description
        results = lib.search_tasks("question")
        assert len(results) > 0

        # Search for non-existent term
        results = lib.search_tasks("nonexistent_term_xyz")
        assert len(results) == 0

    def test_register_task(self):
        """Test registering custom tasks."""
        lib = TaskLibrary()

        custom_spec = {
            "task": "custom.task:CustomTask",
            "dataset": {"type": "custom.dataset:CustomDataset"},
            "adapter": {"type": "custom.adapter:CustomAdapter"},
            "metrics": [{"type": "custom.metric:CustomMetric"}],
        }

        lib.register_task(
            "custom_task",
            category="custom",
            description="A custom test task",
            spec=custom_spec,
            tags=["test", "custom"],
        )

        # Verify task was registered
        task = lib.get_task("custom_task")
        assert task is not None
        assert task["category"] == "custom"
        assert task["tags"] == ["test", "custom"]
        assert task["spec"] == custom_spec

        # Verify category was created
        assert "custom" in lib.list_categories()
        assert "custom_task" in lib.get_category_tasks("custom")

    def test_export_import_task(self):
        """Test exporting and importing tasks."""
        lib = TaskLibrary()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            # Export existing task
            lib.export_task("qa_basic", temp_file)

            # Verify file was created and contains valid JSON
            assert os.path.exists(temp_file)
            with open(temp_file, "r") as f:
                exported_spec = json.load(f)

            assert "task" in exported_spec
            assert "dataset" in exported_spec

            # Import as new task
            lib.import_task(
                "imported_qa", temp_file, category="imported", description="Imported QA task"
            )

            # Verify imported task
            imported_task = lib.get_task("imported_qa")
            assert imported_task is not None
            assert imported_task["category"] == "imported"

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_export_nonexistent_task(self):
        """Test exporting non-existent task raises error."""
        lib = TaskLibrary()

        with pytest.raises(ValueError, match="Task nonexistent not found"):
            lib.export_task("nonexistent", "dummy.json")


class TestBenchmarkSuite:
    """Test benchmark suite functionality."""

    def test_suite_creation(self):
        """Test creating benchmark suites."""
        suite = BenchmarkSuite("test_suite", ["qa_basic", "mcqa_standard"])

        assert suite.name == "test_suite"
        assert suite.task_ids == ["qa_basic", "mcqa_standard"]
        assert len(suite.weights) == 2
        assert all(w == 1.0 for w in suite.weights)

    def test_suite_with_weights(self):
        """Test creating suite with custom weights."""
        suite = BenchmarkSuite("weighted_suite", ["task1", "task2"], [0.3, 0.7])

        assert suite.weights == [0.3, 0.7]

    def test_suite_weight_validation(self):
        """Test that suite validates weight count."""
        with pytest.raises(ValueError, match="Number of weights must match"):
            BenchmarkSuite("invalid_suite", ["task1", "task2"], [1.0])

    def test_extract_primary_score(self):
        """Test extracting primary scores from metrics."""
        suite = BenchmarkSuite("test", ["dummy"])

        # Test with accuracy
        metrics = {"accuracy": 0.85, "f1_score": 0.80}
        score = suite._extract_primary_score(metrics)
        assert score == 0.85

        # Test with f1_score when no accuracy
        metrics = {"f1_score": 0.75, "other": 0.90}
        score = suite._extract_primary_score(metrics)
        assert score == 0.75

        # Test with no priority metrics
        metrics = {"custom_metric": 0.60}
        score = suite._extract_primary_score(metrics)
        assert score == 0.60

        # Test with no numeric metrics
        metrics = {"status": "completed"}
        score = suite._extract_primary_score(metrics)
        assert score == 0.0


class TestGlobalFunctions:
    """Test global library functions."""

    def test_get_task_library(self):
        """Test getting global library instance."""
        lib1 = get_task_library()
        lib2 = get_task_library()

        # Should return the same instance
        assert lib1 is lib2

    def test_list_available_tasks(self):
        """Test listing all available tasks."""
        tasks = list_available_tasks()
        assert isinstance(tasks, list)
        assert len(tasks) > 0
        assert "qa_basic" in tasks

    def test_get_task_info(self):
        """Test getting task information."""
        info = get_task_info("qa_basic")
        assert info is not None
        assert info["id"] == "qa_basic"

        # Test non-existent task
        info = get_task_info("nonexistent")
        assert info is None


class TestPreDefinedSuites:
    """Test pre-defined benchmark suites."""

    def test_nlp_basics_suite(self):
        """Test NLP basics suite configuration."""
        from openeval.library import NLP_BASICS_SUITE

        assert NLP_BASICS_SUITE.name == "nlp_basics"
        assert len(NLP_BASICS_SUITE.task_ids) == 3
        assert sum(NLP_BASICS_SUITE.weights) == pytest.approx(1.0)

    def test_code_suite(self):
        """Test code generation suite configuration."""
        from openeval.library import CODE_SUITE

        assert CODE_SUITE.name == "code_generation"
        assert "code_humaneval" in CODE_SUITE.task_ids
        assert sum(CODE_SUITE.weights) == pytest.approx(1.0)

    def test_comprehensive_suite(self):
        """Test comprehensive suite configuration."""
        from openeval.library import COMPREHENSIVE_SUITE

        assert COMPREHENSIVE_SUITE.name == "comprehensive"
        assert len(COMPREHENSIVE_SUITE.task_ids) >= 3
        assert sum(COMPREHENSIVE_SUITE.weights) == pytest.approx(1.0)
