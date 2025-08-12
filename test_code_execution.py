#!/usr/bin/env python3
"""Test script for code execution evaluation."""

import sys
sys.path.insert(0, 'src')

from openeval.metrics.code_execution import HumanEvalMetric
from openeval.datasets.code import HumanEvalDataset, HumanEvalExample
from openeval.tasks.code import HumanEvalTask

def test_code_execution():
    """Test full code execution pipeline."""
    print("Testing code execution evaluation...")
    
    # Test imports
    try:
        print("✅ All imports successful")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test dataset loading
    try:
        dataset = HumanEvalDataset(path="examples/code_toy.jsonl")
        examples = list(dataset)
        print(f"✅ Loaded {len(examples)} examples from dataset")
        
        # Print first example
        if examples:
            ex = examples[0]
            if isinstance(ex, HumanEvalExample):
                print(f"   Example 1: {ex.prompt[:80]}...")
                print(f"   Test cases: {len(ex.test_cases)}")
            else:
                print(f"   Example 1: ID={ex.id}")
                print(f"   Input: {str(ex.input)[:80]}...")
                
    except Exception as e:
        print(f"❌ Dataset loading error: {e}")
        return False
    
    # Test metric
    try:
        metric = HumanEvalMetric()
        
        # Test with some sample code completions
        predictions = [
            "    return a + b",  # Correct for first example
            "    return s[::-1]",  # Correct for second example  
            "    return n % 3 == 0"  # Wrong for third example (should be % 2)
        ]
        
        references = [
            "assert add_two_numbers(2, 3) == 5\nassert add_two_numbers(-1, 1) == 0",
            "assert reverse_string('hello') == 'olleh'\nassert reverse_string('') == ''",
            "assert is_even(2) == True\nassert is_even(3) == False"
        ]
        
        result = metric.compute(predictions, references)
        print(f"✅ Metric computation successful!")
        print(f"   Accuracy: {result['accuracy']:.2f}")
        print(f"   pass@1: {result['pass_at_k'].get('pass@1', 0):.2f}")
        print(f"   Correct: {result['total_correct']}/{result['total_tests']}")
        
        if result['errors']:
            print(f"   Errors: {len(result['errors'])}")
            for error in result['errors'][:2]:  # Show first 2 errors
                print(f"     - {error}")
        
    except Exception as e:
        print(f"❌ Metric error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    if test_code_execution():
        print("\n🎉 All code execution tests passed!")
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)
