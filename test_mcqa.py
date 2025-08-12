#!/usr/bin/env python3
"""Test script for MCQA log-likelihood evaluation."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from openeval.spec import Spec
from openeval.core import run_evaluation

def test_mcqa_loglik():
    """Test MCQA with log-likelihood evaluation."""
    print("Testing MCQA log-likelihood evaluation...")
    
    # Load spec
    with open('examples/mcqa_loglik_spec.json') as f:
        spec_data = json.load(f)
    
    spec = Spec(**spec_data)
    print(f"Loaded spec: {spec}")
    
    # Run evaluation
    try:
        result = run_evaluation(spec)
        print(f"Evaluation completed successfully!")
        print(f"Result keys: {result.keys()}")
        
        if 'results' in result:
            print(f"Number of results: {len(result['results'])}")
            for i, res in enumerate(result['results'][:3]):  # Show first 3
                print(f"Result {i+1}: {res}")
        
        if 'metrics' in result:
            print(f"Metrics: {result['metrics']}")
            
        return result
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_mcqa_loglik()
    if result:
        print("✅ MCQA log-likelihood evaluation test passed!")
    else:
        print("❌ MCQA log-likelihood evaluation test failed!")
        sys.exit(1)
