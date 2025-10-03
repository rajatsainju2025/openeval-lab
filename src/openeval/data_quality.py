"""Data quality assessment and validation for OpenEval Lab datasets."""

import json
import re
import statistics
from typing import List, Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from pathlib import Path

from .core import Dataset, Example
from .logging import get_logger


@dataclass
class QualityMetric:
    """A single data quality metric."""

    name: str
    value: float
    description: str
    threshold: Optional[float] = None
    passed: Optional[bool] = None

    def __post_init__(self):
        if self.threshold is not None and self.passed is None:
            self.passed = self.value >= self.threshold


@dataclass
class QualityReport:
    """Comprehensive data quality assessment report."""

    dataset_name: str
    sample_count: int
    metrics: List[QualityMetric]
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    overall_score: float = 0.0

    def __post_init__(self):
        # Calculate overall score as average of passed metrics
        if self.metrics:
            passed_metrics = [m for m in self.metrics if m.passed is not None]
            if passed_metrics:
                self.overall_score = sum(1.0 if m.passed else 0.0 for m in passed_metrics) / len(
                    passed_metrics
                )


class DataQualityAssessor:
    """Comprehensive data quality assessment for datasets."""

    def __init__(self):
        """Initialize data quality assessor."""
        self.logger = get_logger()

        # Quality thresholds
        self.thresholds = {
            "completeness": 0.95,  # 95% of samples should be complete
            "uniqueness": 0.90,  # 90% of samples should be unique
            "consistency": 0.85,  # 85% consistency in format/structure
            "relevance": 0.80,  # 80% relevance based on heuristics
            "readability": 0.75,  # 75% readability score
            "balance": 0.20,  # At least 20% representation for minority classes
        }

    def assess_dataset(self, dataset: Dataset, sample_limit: Optional[int] = 1000) -> QualityReport:
        """Perform comprehensive quality assessment on a dataset."""
        self.logger.info(f"Starting quality assessment for dataset: {dataset.__class__.__name__}")

        # Load samples
        samples = list(dataset)
        if sample_limit and len(samples) > sample_limit:
            samples = samples[:sample_limit]
            self.logger.info(f"Limited assessment to {sample_limit} samples")

        metrics = []
        issues = []
        recommendations = []

        # Run all quality checks
        metrics.extend(self._assess_completeness(samples, issues, recommendations))
        metrics.extend(self._assess_uniqueness(samples, issues, recommendations))
        metrics.extend(self._assess_consistency(samples, issues, recommendations))
        metrics.extend(self._assess_relevance(samples, issues, recommendations))
        metrics.extend(self._assess_readability(samples, issues, recommendations))
        metrics.extend(self._assess_balance(samples, issues, recommendations))
        metrics.extend(self._assess_text_quality(samples, issues, recommendations))

        report = QualityReport(
            dataset_name=dataset.__class__.__name__,
            sample_count=len(samples),
            metrics=metrics,
            issues=issues,
            recommendations=recommendations,
        )

        self.logger.info(f"Quality assessment completed. Overall score: {report.overall_score:.2f}")
        return report

    def _assess_completeness(
        self, samples: List[Example], issues: List[str], recommendations: List[str]
    ) -> List[QualityMetric]:
        """Assess data completeness."""
        complete_samples = 0

        for sample in samples:
            is_complete = True

            # Check required fields
            if not sample.id or not sample.input or not sample.reference:
                is_complete = False

            # Check for empty or whitespace-only content
            if isinstance(sample.input, str) and not sample.input.strip():
                is_complete = False

            if isinstance(sample.reference, str) and not sample.reference.strip():
                is_complete = False

            if is_complete:
                complete_samples += 1

        completeness = complete_samples / len(samples) if samples else 0.0

        if completeness < self.thresholds["completeness"]:
            missing_count = len(samples) - complete_samples
            issues.append(
                f"Found {missing_count} incomplete samples ({missing_count/len(samples):.1%})"
            )
            recommendations.append("Remove or fix incomplete samples with missing required fields")

        return [
            QualityMetric(
                name="completeness",
                value=completeness,
                description="Fraction of samples with all required fields",
                threshold=self.thresholds["completeness"],
            )
        ]

    def _assess_uniqueness(
        self, samples: List[Example], issues: List[str], recommendations: List[str]
    ) -> List[QualityMetric]:
        """Assess data uniqueness."""
        # Check for duplicate inputs
        input_hashes = []
        for sample in samples:
            if isinstance(sample.input, str):
                input_hash = hash(sample.input.strip().lower())
                input_hashes.append(input_hash)

        unique_inputs = len(set(input_hashes))
        input_uniqueness = unique_inputs / len(input_hashes) if input_hashes else 0.0

        # Check for duplicate input-reference pairs
        pair_hashes = []
        for sample in samples:
            if isinstance(sample.input, str) and isinstance(sample.reference, str):
                pair_hash = hash((sample.input.strip().lower(), sample.reference.strip().lower()))
                pair_hashes.append(pair_hash)

        unique_pairs = len(set(pair_hashes))
        pair_uniqueness = unique_pairs / len(pair_hashes) if pair_hashes else 0.0

        overall_uniqueness = (input_uniqueness + pair_uniqueness) / 2

        if overall_uniqueness < self.thresholds["uniqueness"]:
            duplicate_inputs = len(input_hashes) - unique_inputs
            duplicate_pairs = len(pair_hashes) - unique_pairs
            issues.append(
                f"Found {duplicate_inputs} duplicate inputs and {duplicate_pairs} duplicate pairs"
            )
            recommendations.append("Remove duplicate samples to improve dataset quality")

        return [
            QualityMetric(
                name="input_uniqueness",
                value=input_uniqueness,
                description="Fraction of unique inputs",
                threshold=self.thresholds["uniqueness"],
            ),
            QualityMetric(
                name="pair_uniqueness",
                value=pair_uniqueness,
                description="Fraction of unique input-reference pairs",
                threshold=self.thresholds["uniqueness"],
            ),
        ]

    def _assess_consistency(
        self, samples: List[Example], issues: List[str], recommendations: List[str]
    ) -> List[QualityMetric]:
        """Assess data consistency."""
        metrics = []

        # Check input length consistency
        input_lengths = []
        for sample in samples:
            if isinstance(sample.input, str):
                input_lengths.append(len(sample.input))

        if input_lengths:
            mean_length = statistics.mean(input_lengths)
            std_length = statistics.stdev(input_lengths) if len(input_lengths) > 1 else 0
            cv_length = std_length / mean_length if mean_length > 0 else 0

            # Lower coefficient of variation indicates higher consistency
            length_consistency = max(0, 1 - cv_length)

            metrics.append(
                QualityMetric(
                    name="input_length_consistency",
                    value=length_consistency,
                    description="Consistency of input lengths (1 - coefficient of variation)",
                    threshold=self.thresholds["consistency"],
                )
            )

        # Check format consistency
        format_patterns = Counter()
        for sample in samples:
            if isinstance(sample.input, str):
                # Simple pattern detection
                has_question_mark = "?" in sample.input
                has_punctuation = any(p in sample.input for p in ".!?")
                starts_with_capital = sample.input.strip() and sample.input.strip()[0].isupper()

                pattern = (has_question_mark, has_punctuation, starts_with_capital)
                format_patterns[pattern] += 1

        if format_patterns:
            most_common_pattern_count = format_patterns.most_common(1)[0][1]
            format_consistency = most_common_pattern_count / len(samples)

            metrics.append(
                QualityMetric(
                    name="format_consistency",
                    value=format_consistency,
                    description="Fraction of samples following the most common format pattern",
                    threshold=self.thresholds["consistency"],
                )
            )

            if format_consistency < self.thresholds["consistency"]:
                issues.append("Inconsistent formatting detected across samples")
                recommendations.append("Standardize input formatting (capitalization, punctuation)")

        return metrics

    def _assess_relevance(
        self, samples: List[Example], issues: List[str], recommendations: List[str]
    ) -> List[QualityMetric]:
        """Assess data relevance using heuristics."""
        relevant_samples = 0

        for sample in samples:
            is_relevant = True

            if isinstance(sample.input, str) and isinstance(sample.reference, str):
                input_text = sample.input.lower()
                reference_text = sample.reference.lower()

                # Check for obvious mismatches
                if len(reference_text) > len(input_text) * 10:
                    is_relevant = False  # Reference way too long

                if len(input_text) < 3 or len(reference_text) < 1:
                    is_relevant = False  # Too short to be meaningful

                # Check for common keywords/patterns that indicate good QA pairs
                question_words = ["what", "how", "when", "where", "who", "why", "which"]
                has_question_word = any(word in input_text for word in question_words)
                has_question_mark = "?" in input_text

                # If it looks like a question, reference shouldn't be another question
                if (has_question_word or has_question_mark) and "?" in reference_text:
                    is_relevant = False

            if is_relevant:
                relevant_samples += 1

        relevance = relevant_samples / len(samples) if samples else 0.0

        if relevance < self.thresholds["relevance"]:
            irrelevant_count = len(samples) - relevant_samples
            issues.append(f"Found {irrelevant_count} potentially irrelevant samples")
            recommendations.append(
                "Review and remove samples with mismatched input-reference pairs"
            )

        return [
            QualityMetric(
                name="relevance",
                value=relevance,
                description="Fraction of samples with relevant input-reference relationships",
                threshold=self.thresholds["relevance"],
            )
        ]

    def _assess_readability(
        self, samples: List[Example], issues: List[str], recommendations: List[str]
    ) -> List[QualityMetric]:
        """Assess text readability."""
        readability_scores = []

        for sample in samples:
            if isinstance(sample.input, str):
                score = self._calculate_readability_score(sample.input)
                readability_scores.append(score)

        if readability_scores:
            avg_readability = statistics.mean(readability_scores)

            if avg_readability < self.thresholds["readability"]:
                issues.append(f"Low average readability score: {avg_readability:.2f}")
                recommendations.append("Improve text clarity and reduce complex sentences")

            return [
                QualityMetric(
                    name="readability",
                    value=avg_readability,
                    description="Average readability score (0-1, higher is better)",
                    threshold=self.thresholds["readability"],
                )
            ]

        return []

    def _calculate_readability_score(self, text: str) -> float:
        """Calculate a simple readability score."""
        if not text.strip():
            return 0.0

        # Simple readability metrics
        sentences = len(re.split(r"[.!?]+", text))
        words = len(text.split())
        characters = len(text.replace(" ", ""))

        if sentences == 0 or words == 0:
            return 0.0

        # Average words per sentence (lower is better)
        avg_words_per_sentence = words / sentences

        # Average characters per word (lower is better)
        avg_chars_per_word = characters / words

        # Normalize to 0-1 scale (higher is better)
        sentence_score = max(0, 1 - (avg_words_per_sentence - 10) / 20)  # Penalty after 10 words
        word_score = max(0, 1 - (avg_chars_per_word - 5) / 10)  # Penalty after 5 chars

        return (sentence_score + word_score) / 2

    def _assess_balance(
        self, samples: List[Example], issues: List[str], recommendations: List[str]
    ) -> List[QualityMetric]:
        """Assess dataset balance."""
        # Simple balance check based on reference length categories
        reference_categories = defaultdict(int)

        for sample in samples:
            if isinstance(sample.reference, str):
                ref_len = len(sample.reference.strip())
                if ref_len < 10:
                    category = "short"
                elif ref_len < 50:
                    category = "medium"
                else:
                    category = "long"
                reference_categories[category] += 1

        if reference_categories:
            total_samples = sum(reference_categories.values())
            proportions = {
                cat: count / total_samples for cat, count in reference_categories.items()
            }

            # Check if any category has very low representation
            min_proportion = min(proportions.values())
            balance_score = min_proportion / self.thresholds["balance"]
            balance_score = min(1.0, balance_score)  # Cap at 1.0

            if min_proportion < self.thresholds["balance"]:
                issues.append(f"Imbalanced dataset - minimum category: {min_proportion:.1%}")
                recommendations.append("Consider balancing reference length categories")

            return [
                QualityMetric(
                    name="reference_length_balance",
                    value=balance_score,
                    description="Balance of reference length categories",
                    threshold=1.0,  # This is already normalized
                )
            ]

        return []

    def _assess_text_quality(
        self, samples: List[Example], issues: List[str], recommendations: List[str]
    ) -> List[QualityMetric]:
        """Assess general text quality."""
        quality_issues = 0

        for sample in samples:
            has_issues = False

            # Check input quality
            if isinstance(sample.input, str):
                input_text = sample.input.strip()

                # Check for excessive repetition
                words = input_text.lower().split()
                if len(words) > 5:
                    word_counts = Counter(words)
                    most_common_count = word_counts.most_common(1)[0][1]
                    if most_common_count > len(words) * 0.3:  # 30% repetition
                        has_issues = True

                # Check for encoding issues
                if "�" in input_text or "\x00" in input_text:
                    has_issues = True

                # Check for excessive punctuation
                punct_ratio = sum(
                    1 for c in input_text if not c.isalnum() and not c.isspace()
                ) / len(input_text)
                if punct_ratio > 0.3:  # 30% punctuation
                    has_issues = True

            # Similar checks for reference
            if isinstance(sample.reference, str):
                ref_text = sample.reference.strip()

                if "�" in ref_text or "\x00" in ref_text:
                    has_issues = True

            if has_issues:
                quality_issues += 1

        text_quality = 1 - (quality_issues / len(samples)) if samples else 1.0

        if text_quality < 0.9:  # 90% threshold for text quality
            issues.append(f"Found text quality issues in {quality_issues} samples")
            recommendations.append(
                "Clean text data to remove encoding issues and excessive repetition"
            )

        return [
            QualityMetric(
                name="text_quality",
                value=text_quality,
                description="Fraction of samples without text quality issues",
                threshold=0.9,
            )
        ]

    def generate_report(self, quality_report: QualityReport) -> str:
        """Generate a human-readable quality assessment report."""
        report = ["# Data Quality Assessment Report\n"]

        # Executive summary
        report.append("## Executive Summary\n")
        report.append(f"**Dataset**: {quality_report.dataset_name}")
        report.append(f"**Sample Count**: {quality_report.sample_count:,}")
        report.append(f"**Overall Quality Score**: {quality_report.overall_score:.2f}/1.00")

        # Quality status
        if quality_report.overall_score >= 0.8:
            status = "🟢 **GOOD** - Dataset meets quality standards"
        elif quality_report.overall_score >= 0.6:
            status = "🟡 **FAIR** - Dataset has some quality issues"
        else:
            status = "🔴 **POOR** - Dataset requires significant improvement"

        report.append(f"**Status**: {status}\n")

        # Detailed metrics
        report.append("## Quality Metrics\n")

        for metric in quality_report.metrics:
            status_icon = "✅" if metric.passed else "❌" if metric.passed is False else "ℹ️"
            report.append(f"### {metric.name} {status_icon}")
            report.append(f"**Score**: {metric.value:.3f}")
            report.append(f"**Description**: {metric.description}")

            if metric.threshold is not None:
                report.append(f"**Threshold**: {metric.threshold:.3f}")

            report.append("")

        # Issues found
        if quality_report.issues:
            report.append("## Issues Found\n")
            for issue in quality_report.issues:
                report.append(f"- ⚠️ {issue}")
            report.append("")

        # Recommendations
        if quality_report.recommendations:
            report.append("## Recommendations\n")
            for rec in quality_report.recommendations:
                report.append(f"- 💡 {rec}")
            report.append("")

        return "\n".join(report)

    def save_report(self, quality_report: QualityReport, output_path: Path) -> Path:
        """Save quality assessment report to file."""
        # Generate markdown report
        markdown_report = self.generate_report(quality_report)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(markdown_report)

        # Also save JSON version
        json_path = output_path.with_suffix(".json")
        json_data = {
            "dataset_name": quality_report.dataset_name,
            "sample_count": quality_report.sample_count,
            "overall_score": quality_report.overall_score,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "description": m.description,
                    "threshold": m.threshold,
                    "passed": m.passed,
                }
                for m in quality_report.metrics
            ],
            "issues": quality_report.issues,
            "recommendations": quality_report.recommendations,
        }

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        self.logger.info(f"Quality report saved to {output_path}")
        self.logger.info(f"JSON report saved to {json_path}")

        return output_path
