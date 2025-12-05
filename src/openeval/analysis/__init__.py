"""Analysis module for evaluation results."""

from .comparison import (
    ResultComparer,
    ResultComparison,
    MetricComparison,
    side_by_side_table,
    export_comparison,
    diff_results,
)

__all__ = [
    "ResultComparer",
    "ResultComparison",
    "MetricComparison",
    "side_by_side_table",
    "export_comparison",
    "diff_results",
]
