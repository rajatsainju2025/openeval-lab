"""Example demonstrating OpenEval's profiling utilities.

This example shows how to use the profiling tools to measure and optimize
evaluation performance.
"""

from openeval import profile_time, profile_block, PerformanceTimer


# Example 1: Function-level profiling with decorator
@profile_time
def load_dataset():
    """Simulate loading a dataset."""
    import time

    time.sleep(0.5)
    return ["sample1", "sample2", "sample3"]


@profile_time
def process_samples(samples):
    """Simulate processing samples."""
    import time

    results = []
    for sample in samples:
        time.sleep(0.1)
        results.append(f"processed_{sample}")
    return results


# Example 2: Code block profiling with context manager
def example_with_blocks():
    """Example using profile_block context manager."""

    with profile_block("initialization"):
        import time

        # Simulate initialization
        time.sleep(0.2)
        _ = {"batch_size": 32, "model": "gpt-4"}  # Config loaded

    with profile_block("data loading"):
        # Simulate data loading
        time.sleep(0.3)
        data = list(range(100))

    with profile_block("preprocessing"):
        # Simulate preprocessing
        time.sleep(0.15)
        processed = [x * 2 for x in data]

    return processed


# Example 3: Multi-operation tracking with PerformanceTimer
def example_with_timer():
    """Example using PerformanceTimer for detailed tracking."""
    import time

    timer = PerformanceTimer()

    # Track multiple operations
    timer.start("model_loading")
    time.sleep(0.4)
    timer.stop("model_loading")

    timer.start("tokenization")
    time.sleep(0.2)
    timer.stop("tokenization")

    timer.start("inference")
    for i in range(3):
        time.sleep(0.1)
    timer.stop("inference")

    timer.start("postprocessing")
    time.sleep(0.15)
    timer.stop("postprocessing")

    # Print formatted report
    timer.report()

    # Access timings programmatically
    if timer.timings["inference"] > 0.3:
        print("⚠️  Inference is slow, consider optimizing!")


# Example 4: Real evaluation scenario
@profile_time
def evaluate_model():
    """Simulate a full evaluation workflow."""
    timer = PerformanceTimer()

    # Step 1: Load data
    timer.start("data_loading")
    with profile_block("dataset initialization"):
        samples = load_dataset()
    timer.stop("data_loading")

    # Step 2: Process
    timer.start("processing")
    results = process_samples(samples)
    timer.stop("processing")

    # Step 3: Compute metrics
    timer.start("metrics")
    import time

    time.sleep(0.1)
    accuracy = 0.95
    timer.stop("metrics")

    timer.report()
    return {"accuracy": accuracy, "num_samples": len(results)}


if __name__ == "__main__":
    print("=" * 60)
    print("OpenEval Profiling Examples")
    print("=" * 60)

    print("\n📊 Example 1: Function-level profiling")
    print("-" * 60)
    data = load_dataset()
    results = process_samples(data)

    print("\n📊 Example 2: Code block profiling")
    print("-" * 60)
    example_with_blocks()

    print("\n📊 Example 3: Performance timer")
    print("-" * 60)
    example_with_timer()

    print("\n📊 Example 4: Full evaluation workflow")
    print("-" * 60)
    results = evaluate_model()
    print(f"\nFinal results: {results}")

    print("\n✅ All examples completed successfully!")
