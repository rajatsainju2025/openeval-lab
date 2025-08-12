#!/usr/bin/env python3
"""Test script for curated task library."""

import sys
sys.path.insert(0, 'src')

def test_library():
    """Test task library functionality."""
    print("Testing curated task library...")
    
    # Test imports
    try:
        from openeval.library import (
            get_task_library,
            list_available_tasks,
            get_task_info,
            BenchmarkSuite
        )
        print("✅ All imports successful")
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test library access
    try:
        lib = get_task_library()
        print("✅ Task library initialized")
        
        # List available tasks
        tasks = list_available_tasks()
        print(f"✅ Found {len(tasks)} available tasks:")
        for task_id in tasks[:5]:  # Show first 5
            print(f"   - {task_id}")
        
        if len(tasks) > 5:
            print(f"   ... and {len(tasks) - 5} more")
            
    except Exception as e:
        print(f"❌ Library access error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test task categories
    try:
        categories = lib.list_categories()
        print(f"✅ Found {len(categories)} categories:")
        for cat in categories:
            cat_tasks = lib.get_category_tasks(cat)
            print(f"   - {cat}: {len(cat_tasks)} tasks")
            
    except Exception as e:
        print(f"❌ Category listing error: {e}")
        return False
    
    # Test specific task info
    try:
        task_info = get_task_info("qa_basic")
        if task_info:
            print("✅ Retrieved qa_basic task info:")
            print(f"   Description: {task_info['description']}")
            print(f"   Category: {task_info['category']}")
        else:
            print("❌ qa_basic task not found")
            return False
            
    except Exception as e:
        print(f"❌ Task info error: {e}")
        return False
    
    # Test search functionality
    try:
        search_results = lib.search_tasks("code")
        print(f"✅ Search for 'code' found {len(search_results)} tasks")
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return False
    
    # Test benchmark suite
    try:
        suite = BenchmarkSuite("test_suite", ["qa_basic", "mcqa_standard"])
        print(f"✅ Created benchmark suite: {suite.name}")
        print(f"   Tasks: {suite.task_ids}")
        print(f"   Weights: {suite.weights}")
        
    except Exception as e:
        print(f"❌ Benchmark suite error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if test_library():
        print("\n🎉 All task library tests passed!")
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)
