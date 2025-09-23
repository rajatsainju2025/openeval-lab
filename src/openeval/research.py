from __future__ import annotations

"""
Research Integration Module for OpenEval Lab.

This module provides integration with academic research, including paper retrieval, 
benchmark comparison, and research insights. It helps connect evaluation tasks with 
the latest research findings and state-of-the-art benchmarks.

The module includes:
- Research paper discovery and filtering
- Benchmark result tracking and comparison
- Citation analysis and leaderboard integration
- Research methodology integration

Examples:
    Basic paper search and methodology integration:
    ```python
    from openeval.research import ResearchIntegrator
    
    # Search for relevant papers on evaluation methods
    integrator = ResearchIntegrator()
    papers = integrator.search_relevant_papers("few-shot evaluation methods")
    
    # Compare model against published benchmarks
    benchmarks = integrator.get_benchmark_results("gpt-4", "mmlu")
    ```
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple
import json
import requests
from pathlib import Path
import time
from datetime import datetime, timedelta

from ..core import Task, Dataset, Adapter, Metric


@dataclass
class ResearchPaper:
    """
    Represents a research paper with evaluation relevance.
    
    This class encapsulates metadata about academic research papers
    related to AI evaluation, allowing for tracking and filtering
    of relevant literature.
    
    Attributes:
        title: The title of the research paper
        authors: List of author names
        abstract: Summary text of the paper
        url: URL to access the full paper
        venue: Publication venue (conference, journal, preprint server)
        year: Publication year
        topics: List of relevant topics/keywords
        relevance_score: Float between 0-1 indicating relevance to evaluation
        citation_count: Optional number of citations
    """

    title: str
    authors: List[str]
    abstract: str
    url: str
    venue: str
    year: int
    topics: List[str]
    relevance_score: float
    citation_count: Optional[int] = None

    def __post_init__(self) -> None:
        """Initialize with empty topics list if none provided."""
        if not self.topics:
            self.topics = []


@dataclass
class BenchmarkResult:
    """
    Result from a benchmark evaluation.
    
    This class stores the results of model evaluations on standard
    benchmarks, facilitating comparison between different models
    and tracking of performance improvements.
    
    Attributes:
        benchmark_name: Name of the benchmark (e.g., "MMLU", "HellaSwag")
        model_name: Name of the evaluated model
        metric_name: Name of the evaluation metric used
        score: Numeric score achieved on the benchmark
        confidence_interval: Optional tuple of (lower, upper) bounds
        sample_size: Optional number of examples in the evaluation
        date_evaluated: Optional ISO-format date of evaluation
    """

    benchmark_name: str
    model_name: str
    metric_name: str
    score: float
    confidence_interval: Optional[Tuple[float, float]] = None
    sample_size: Optional[int] = None
    date_evaluated: Optional[str] = None

    def __post_init__(self) -> None:
        """Set date_evaluated to current time if not provided."""
        if self.date_evaluated is None:
            self.date_evaluated = datetime.utcnow().isoformat()


class ResearchIntegrator:
    """
    Integrates latest research findings into evaluation workflows.
    
    This class provides methods to discover, analyze, and integrate
    research findings into evaluation frameworks. It connects the OpenEval
    system with the academic research community, helping users stay informed
    about the latest evaluation methodologies and benchmark results.
    
    Features:
        - Paper discovery via semantic search
        - Benchmark result aggregation and comparison
        - Research methodology integration
        - Citation analysis and leaderboard tracking
    
    The class maintains caches of research papers and benchmark results
    to minimize redundant API calls to external services.
    """

    def __init__(self) -> None:
        """
        Initialize the ResearchIntegrator with empty caches.
        
        Sets up in-memory caches for papers and benchmark results with
        a default expiry time of 24 hours to balance freshness with
        API usage efficiency.
        """
        self.papers_cache: Dict[str, List[ResearchPaper]] = {}
        self.benchmarks_cache: Dict[str, List[BenchmarkResult]] = {}
        self.cache_expiry = timedelta(hours=24)

    def search_relevant_papers(self, query: str, max_results: int = 10) -> List[ResearchPaper]:
        """
        Search for relevant research papers using semantic search.
        
        This method retrieves research papers related to the provided query using
        semantic search techniques. It integrates with academic APIs like 
        Semantic Scholar, arXiv, or CrossRef to find the most relevant papers.
        
        Args:
            query: The search query (e.g., "few-shot evaluation methods")
            max_results: Maximum number of papers to return
            
        Returns:
            A list of ResearchPaper objects, sorted by relevance
            
        Note:
            Results are cached for performance. Use refresh_cache=True
            in future implementations to force a fresh search.
        """
        # Mock implementation - in practice, would call real APIs
        mock_papers = [
            ResearchPaper(
                title="Evaluating Large Language Models: A Comprehensive Survey",
                authors=["Yejin Bang", "Samuel Cahyawijaya", "Nayeon Lee"],
                abstract="Comprehensive survey of LLM evaluation methodologies...",
                url="https://arxiv.org/abs/2310.19736",
                venue="arXiv",
                year=2023,
                topics=["evaluation", "survey", "methodology"],
                relevance_score=0.95
            ),
            ResearchPaper(
                title="HELM: Holistic Evaluation of Language Models",
                authors=["Perez et al."],
                abstract="Multi-metric evaluation framework for LLMs...",
                url="https://arxiv.org/abs/2211.09110",
                venue="TMLR",
                year=2022,
                topics=["evaluation", "metrics", "holistic"],
                relevance_score=0.92
            )
        ]

        # Filter by query relevance - optimized with pre-tokenization
        relevant_papers = []
        query_tokens = set(query.lower().split())

        for paper in mock_papers:
            # Pre-compute searchable text once per paper
            searchable_text = (
                paper.title.lower() + " " +
                paper.abstract.lower() + " " +
                ' '.join(paper.topics).lower()
            )
            
            # Use set intersection for O(1) average-case lookup
            if query_tokens & set(searchable_text.split()):
                relevant_papers.append(paper)

        return relevant_papers[:max_results]

    def get_latest_benchmarks(self, model_name: Optional[str] = None) -> List[BenchmarkResult]:
        """
        Retrieve latest benchmark results for models.

        Gets the most recent benchmark results, optionally filtered by model name.
        These results can be used to compare model performance across standard
        evaluation benchmarks and track progress over time.
        
        Args:
            model_name: Optional name of a specific model to filter by
                        (e.g., "gpt-4", "claude-3-opus")
            
        Returns:
            A list of BenchmarkResult objects containing performance metrics
            
        Note:
            In production, this would integrate with leaderboard APIs like 
            Hugging Face, OpenCompass, etc.
            
        Example:
            >>> integrator = ResearchIntegrator()
            >>> results = integrator.get_latest_benchmarks("gpt-4")
            >>> for result in results:
            ...     print(f"{result.benchmark_name}: {result.score:.2f}")
            MMLU: 0.87
            GSM8K: 0.92
        """
        # Mock implementation
        mock_results = [
            BenchmarkResult(
                benchmark_name="MMLU",
                model_name="GPT-4",
                metric_name="accuracy",
                score=0.87,
                confidence_interval=(0.85, 0.89),
                sample_size=14042
            ),
            BenchmarkResult(
                benchmark_name="GSM8K",
                model_name="GPT-4",
                metric_name="accuracy",
                score=0.92,
                confidence_interval=(0.90, 0.94),
                sample_size=1319
            )
        ]

        if model_name:
            mock_results = [r for r in mock_results if model_name.lower() in r.model_name.lower()]

        return mock_results

    def suggest_evaluation_methodology(self, task_type: str, model_type: str) -> Dict[str, Any]:
        """
        Suggest evaluation methodology based on latest research.
        
        Provides recommendations for metrics, statistical tests, and bias checks
        appropriate for evaluating specific types of tasks and models based on
        recent research findings.
        
        Args:
            task_type: The type of task being evaluated (e.g., "qa", "code_generation")
            model_type: The type or family of model being evaluated (e.g., "llm", "multimodal")
            
        Returns:
            A dictionary containing recommended evaluation methodology:
            - recommended_metrics: List of appropriate metrics
            - statistical_tests: Statistical methods for significance testing
            - bias_checks: Recommended bias and robustness checks
            - research_basis: Brief explanation of research supporting these recommendations
            
        Example:
            >>> integrator = ResearchIntegrator()
            >>> method = integrator.suggest_evaluation_methodology("qa", "llm")
            >>> print(f"Recommended metrics: {', '.join(method['recommended_metrics'])}")
            Recommended metrics: exact_match, f1, semantic_similarity
        """
        suggestions = {
            "qa": {
                "recommended_metrics": ["exact_match", "f1", "semantic_similarity"],
                "statistical_tests": ["bootstrap_ci", "mcnemar_test"],
                "bias_checks": ["positional_bias", "prompt_sensitivity"],
                "research_basis": "Recent studies show exact_match alone insufficient for QA evaluation"
            },
            "code_generation": {
                "recommended_metrics": ["pass@k", "codebleu", "compilation_rate"],
                "statistical_tests": ["paired_bootstrap"],
                "bias_checks": ["dataset_contamination"],
                "research_basis": "Code evaluation requires specialized metrics per HumanEval paper"
            },
            "multimodal": {
                "recommended_metrics": ["llm_judge", "clip_score", "human_preference"],
                "statistical_tests": ["bootstrap_ci"],
                "bias_checks": ["modality_bias"],
                "research_basis": "Multimodal evaluation needs cross-modal alignment metrics"
            }
        }

        return suggestions.get(task_type, {
            "recommended_metrics": ["exact_match", "llm_judge"],
            "statistical_tests": ["bootstrap_ci"],
            "bias_checks": ["general_bias"],
            "research_basis": "General evaluation best practices"
        })

    def validate_against_sota(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate evaluation results against state-of-the-art benchmarks.
        
        Compares evaluation results with state-of-the-art benchmarks to provide
        context on model performance relative to the current research frontier.
        This helps identify areas for improvement and contextualize results.
        
        Args:
            results: Dictionary containing evaluation results with keys:
                   - adapter: Information about the model adapter
                   - metrics: Dictionary mapping metric names to scores
                   
        Returns:
            A validation report dictionary with:
            - comparison_results: List of metric comparisons with SOTA
            - recommendations: List of suggested improvements based on comparisons
            - research_gaps: Identified areas where more research is needed
            
        Example:
            >>> results = {
            ...     "adapter": {"model": "gpt-4"},
            ...     "metrics": {"accuracy": 0.82, "f1": 0.78}
            ... }
            >>> report = integrator.validate_against_sota(results)
            >>> if report["recommendations"]:
            ...     print("Recommendations:", report["recommendations"][0])
        """
        validation_report = {
            "comparison_results": [],
            "recommendations": [],
            "research_gaps": []
        }

        # Get relevant benchmarks
        model_name = results.get("adapter", {}).get("model", "unknown")
        benchmarks = self.get_latest_benchmarks(model_name)

        for benchmark in benchmarks:
            metric_name = benchmark.metric_name
            if metric_name in results.get("metrics", {}):
                our_score = results["metrics"][metric_name]
                sota_score = benchmark.score

                comparison = {
                    "benchmark": benchmark.benchmark_name,
                    "metric": metric_name,
                    "our_score": our_score,
                    "sota_score": sota_score,
                    "difference": our_score - sota_score,
                    "within_ci": (benchmark.confidence_interval and
                                benchmark.confidence_interval[0] <= our_score <= benchmark.confidence_interval[1])
                }
                validation_report["comparison_results"].append(comparison)

                # Generate recommendations
                if our_score < sota_score * 0.9:  # More than 10% below SOTA
                    validation_report["recommendations"].append(
                        f"Performance significantly below SOTA on {benchmark.benchmark_name}. "
                        "Consider model updates or hyperparameter tuning."
                    )

        return validation_report

    def get_research_driven_improvements(self) -> List[str]:
        """
        Get list of potential improvements based on recent research.
        
        Returns improvement suggestions derived from recent academic research
        in AI evaluation. These suggestions can guide development priorities
        and ensure the evaluation framework stays current with best practices.
        
        Returns:
            A list of improvement suggestions with explanations
            
        Example:
            >>> suggestions = integrator.get_research_driven_improvements()
            >>> for suggestion in suggestions[:2]:
            ...     print(f"- {suggestion}")
            - Implement conformal prediction for uncertainty quantification
            - Add fairness metrics based on recent fairness in ML research
        """
        improvements = [
            "Implement conformal prediction for uncertainty quantification",
            "Add fairness metrics based on recent fairness in ML research",
            "Integrate human-AI collaborative evaluation methods",
            "Support federated evaluation for privacy-preserving assessment",
            "Add multi-modal evaluation capabilities",
            "Implement adaptive evaluation based on model capabilities",
            "Add robustness testing against adversarial inputs",
            "Support evaluation of agent-based systems",
            "Integrate with external benchmark APIs for continuous validation",
            "Add temporal evaluation for model degradation detection"
        ]

        return improvements

    def export_research_summary(self, output_path: str) -> None:
        """
        Export a summary of integrated research findings.
        
        Creates a JSON file with a comprehensive summary of research findings,
        including methodology recommendations, benchmark results, and suggested
        improvements. This can be used for reporting, documentation, or tracking
        research integration over time.
        
        Args:
            output_path: Path where the JSON summary should be saved
            
        Returns:
            None
            
        Example:
            >>> integrator = ResearchIntegrator()
            >>> integrator.export_research_summary("research_summary_2025.json")
            >>> # Creates a JSON file with research insights and recommendations
        """
        summary = {
            "generated_at": datetime.utcnow().isoformat(),
            "research_papers": len(self.papers_cache),
            "benchmark_results": len(self.benchmarks_cache),
            "suggested_improvements": self.get_research_driven_improvements(),
            "methodology_recommendations": {
                "qa": self.suggest_evaluation_methodology("qa", "general"),
                "code": self.suggest_evaluation_methodology("code_generation", "general"),
                "multimodal": self.suggest_evaluation_methodology("multimodal", "general")
            }
        }

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
