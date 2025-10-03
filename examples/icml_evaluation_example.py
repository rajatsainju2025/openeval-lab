"""
Example script demonstrating ICML-standard experimental evaluation.

This script shows how to run comprehensive experiments with statistical analysis,
bias detection, and reproducibility tracking following ICML standards.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openeval.icml_framework import (
    ICMLExperimentRunner,
    ExperimentConfig,
    create_icml_benchmark_suite,
)
from openeval.adapters.echo_adapter import EchoAdapter
from openeval.datasets.jsonl_dataset import JSONLDataset
from openeval.tasks.qa import QATask
from openeval.metrics.exact_match import ExactMatch
from openeval.metrics.token_f1 import TokenF1
from openeval.metrics.statistical import BootstrapAccuracy


def main():
    """Run comprehensive ICML-standard evaluation."""

    print("🔬 ICML-Standard Experimental Evaluation")
    print("=" * 50)

    # Configure experiment
    config = ExperimentConfig(
        name="qa_model_comparison",
        description="Comprehensive comparison of QA models with statistical analysis and bias detection",
        random_seeds=[42, 123, 456, 789, 321],  # Multiple seeds for robustness
        n_bootstrap=1000,
        confidence_level=0.95,
        statistical_tests=["paired_bootstrap", "mcnemar"],
        bias_analysis=True,
        prompt_sensitivity=True,
    )

    print(f"📋 Experiment: {config.name}")
    print(f"📝 Description: {config.description}")
    print(f"🎲 Random seeds: {config.random_seeds}")
    print(f"🔢 Bootstrap samples: {config.n_bootstrap}")
    print(f"📊 Confidence level: {config.confidence_level * 100}%")
    print()

    # Set up task and dataset
    task = QATask()

    # Create simple dataset for demonstration
    dataset_path = Path("examples/qa_toy.jsonl")
    if not dataset_path.exists():
        print("⚠️  Creating example dataset...")
        create_example_dataset(dataset_path)

    dataset = JSONLDataset(dataset_path)

    # Configure adapters for comparison
    adapters = {
        "echo_baseline": EchoAdapter(),
        # Uncomment if you have OpenAI API key
        # "gpt-3.5-turbo": OpenAIAdapter(model="gpt-3.5-turbo"),
    }

    print(f"🤖 Adapters to evaluate: {list(adapters.keys())}")

    # Configure metrics
    metrics = [ExactMatch(), TokenF1(), BootstrapAccuracy(n_bootstrap=100)]  # Statistical metric

    print(f"📏 Metrics: {[m.__class__.__name__ for m in metrics]}")
    print()

    # Initialize experiment runner
    runner = ICMLExperimentRunner(output_dir=Path("experiments"))

    # Run comprehensive evaluation
    print("🚀 Starting comprehensive evaluation...")
    result = runner.run_comprehensive_evaluation(
        config=config, task=task, dataset=dataset, adapters=adapters, metrics=metrics
    )

    # Print results summary
    print("\n" + "=" * 50)
    print("📊 EXPERIMENT RESULTS SUMMARY")
    print("=" * 50)

    print(f"🆔 Experiment ID: {result.experiment_id}")
    print(f"⏰ Timestamp: {result.timestamp}")
    print()

    # Performance results
    print("🎯 PERFORMANCE METRICS")
    print("-" * 30)

    for adapter_name, adapter_results in result.performance_metrics.items():
        print(f"\n🤖 {adapter_name}:")

        if "mean_metrics" in adapter_results:
            for metric_name, mean_value in adapter_results["mean_metrics"].items():
                std_value = adapter_results.get("std_metrics", {}).get(metric_name, 0)
                ci = adapter_results.get("confidence_intervals", {}).get(metric_name, (0, 0))

                print(f"  📏 {metric_name}:")
                print(f"    Mean: {mean_value:.3f} (±{std_value:.3f})")
                print(f"    95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

        print(f"  🎲 Seeds evaluated: {adapter_results.get('n_seeds', 0)}")

    # Statistical significance
    print("\n🔍 STATISTICAL ANALYSIS")
    print("-" * 30)

    if "pairwise_comparisons" in result.statistical_analysis:
        for comparison, analysis in result.statistical_analysis["pairwise_comparisons"].items():
            print(f"\n🆚 {comparison}:")

            for metric_name, stats in analysis.items():
                if isinstance(stats, dict):
                    p_val = stats.get("p_value", 1.0)
                    effect_size = stats.get("effect_size", 0.0)
                    significant = stats.get("significant", False)

                    significance_symbol = "✅" if significant else "❌"

                    print(f"  📏 {metric_name}:")
                    print(f"    p-value: {p_val:.4f} {significance_symbol}")
                    print(f"    Effect size (Cohen's d): {effect_size:.3f}")
                    print(f"    Statistically significant: {significant}")

    # Bias analysis
    print("\n🔍 BIAS ANALYSIS")
    print("-" * 30)

    if result.bias_analysis:
        for bias_type, analysis in result.bias_analysis.items():
            print(f"\n🔍 {bias_type.title()}:")

            if isinstance(analysis, dict):
                for adapter_name, adapter_analysis in analysis.items():
                    if isinstance(adapter_analysis, dict):
                        print(f"  🤖 {adapter_name}:")

                        # Positional bias
                        if "bias_detected" in adapter_analysis:
                            bias_detected = adapter_analysis["bias_detected"]
                            bias_symbol = "⚠️" if bias_detected else "✅"
                            print(f"    Bias detected: {bias_detected} {bias_symbol}")

                        # Sensitivity
                        if "highly_sensitive" in adapter_analysis:
                            sensitive = adapter_analysis["highly_sensitive"]
                            sensitive_symbol = "⚠️" if sensitive else "✅"
                            print(f"    Highly sensitive: {sensitive} {sensitive_symbol}")

    # System information
    print("\n💻 SYSTEM INFORMATION")
    print("-" * 30)
    print(f"🐍 Python: {result.system_info.get('python_version', 'Unknown')}")
    print(f"💻 Platform: {result.system_info.get('platform', 'Unknown')}")
    print(f"⚙️ Processor: {result.system_info.get('processor', 'Unknown')}")

    # Save location
    experiment_dir = runner.output_dir / result.experiment_id
    print(f"\n💾 Results saved to: {experiment_dir}")
    print(f"📄 Summary report: {experiment_dir / 'summary_report.md'}")
    print(f"📊 Full results: {experiment_dir / 'experiment_result.json'}")

    # Reproducibility information
    print("\n🔬 REPRODUCIBILITY")
    print("-" * 30)
    print("✅ Random seeds fixed")
    print("✅ System information recorded")
    print("✅ Model configurations saved")
    print("✅ Statistical analysis included")
    print("✅ Bias analysis performed")

    print("\n🎉 Experiment completed successfully!")
    print(f"📋 Run 'cat {experiment_dir / 'summary_report.md'}' to view the full report")


def create_example_dataset(path: Path):
    """Create example QA dataset for demonstration."""
    import json

    examples = [
        {"input": "What is the capital of France?", "output": "Paris"},
        {"input": "What is 2 + 2?", "output": "4"},
        {"input": "What color is the sky?", "output": "Blue"},
        {"input": "What is the largest planet in our solar system?", "output": "Jupiter"},
        {"input": "How many days are in a week?", "output": "7"},
    ]

    path.parent.mkdir(exist_ok=True)

    with open(path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"✅ Created example dataset with {len(examples)} samples at {path}")


if __name__ == "__main__":
    # Check for ICML benchmark suite
    print("🧪 Available ICML Benchmark Categories:")
    suite = create_icml_benchmark_suite()

    for category, benchmarks in suite.items():
        print(f"  📂 {category}: {len(benchmarks)} benchmarks")
        print(f"    Examples: {', '.join(benchmarks[:3])}")
        if len(benchmarks) > 3:
            print(f"    ... and {len(benchmarks) - 3} more")
        print()

    print("🚀 Starting main evaluation...")
    main()
