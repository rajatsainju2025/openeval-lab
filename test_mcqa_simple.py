#!/usr/bin/env python3
"""Test script for MCQA log-likelihood evaluation."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_mcqa_loglik():
    """Test MCQA with log-likelihood evaluation."""
    print("Testing MCQA log-likelihood evaluation...")
    
    # Test imports
    try:
        from openeval.loglikelihood import MCQATask, MockLogLikelihoodAdapter
        from openeval.datasets.mcqa import MCQADataset, MCQAExample
        from openeval.metrics.loglik_accuracy import LogLikelihoodAccuracy
        print("✅ All imports successful")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test dataset loading
    try:
        dataset = MCQADataset(path="examples/mcqa_toy.jsonl")
        examples = list(dataset)
        print(f"✅ Loaded {len(examples)} examples from dataset")
        
        # Print first example
        if examples:
            ex = examples[0]
            if hasattr(ex, 'question') and hasattr(ex, 'choices'):
                question = getattr(ex, 'question', '')
                choices = getattr(ex, 'choices', [])
                print(f"   Example 1: Q={question[:50]}..., Choices={len(choices)}")
            else:
                print(f"   Example 1: ID={ex.id}, Input={str(ex.input)[:50]}...")
    except Exception as e:
        print(f"❌ Dataset loading error: {e}")
        return False
    
    # Test task and adapter
    try:
        task = MCQATask()
        adapter = MockLogLikelihoodAdapter()
        
        # Test prediction on first example
        if examples:
            result = task.predict(adapter, examples[0])
            print(f"✅ Prediction successful: {result['prediction']}")
            print(f"   Choice: {result['choice']}")
            print(f"   Confidence: {result['confidence']:.3f}")
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test metric
    try:
        metric = LogLikelihoodAccuracy()
        
        # Create mock predictions and references
        predictions = ["A", "B"]
        references = ["A", "A"]
        score = metric.compute(predictions, references)
        print(f"✅ Metric computation successful: {score}")
        
    except Exception as e:
        print(f"❌ Metric error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if test_mcqa_loglik():
        print("\n🎉 All MCQA log-likelihood tests passed!")
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)
