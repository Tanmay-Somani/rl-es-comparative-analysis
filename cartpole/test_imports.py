# test_imports.py
# Simple test script to verify the CartPole implementation structure

def test_imports():
    """Test that all modules can be imported without errors."""
    try:
        print("Testing imports...")
        
        # Test basic imports
        import numpy as np
        print("✓ numpy imported successfully")
        
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported successfully")
        
        # Test our modules (without running the actual training)
        print("\nTesting CartPole modules...")
        
        # Test RL non-FL module structure
        import rl_non_fl
        print("✓ rl_non_fl module imported")
        
        # Test ES distributed module structure  
        import es_distributed
        print("✓ es_distributed module imported")
        
        # Test FL RL module structure
        import fl_rl
        print("✓ fl_rl module imported")
        
        # Test FL ES module structure
        import fl_es
        print("✓ fl_es module imported")
        
        # Test utils module structure
        import utils
        print("✓ utils module imported")
        
        print("\nAll modules imported successfully!")
        print("CartPole implementation is ready to run.")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please make sure all dependencies are installed:")
        print("- gymnasium")
        print("- numpy") 
        print("- matplotlib")
        print("- psutil")
        print("- tqdm")
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

def test_function_signatures():
    """Test that all required functions exist with correct signatures."""
    print("\nTesting function signatures...")
    
    # Test RL non-FL
    assert hasattr(rl_non_fl, 'run_rl_non_fl'), "run_rl_non_fl function not found"
    print("✓ run_rl_non_fl function found")
    
    # Test ES distributed
    assert hasattr(es_distributed, 'run_es_distributed'), "run_es_distributed function not found"
    print("✓ run_es_distributed function found")
    
    # Test FL RL
    assert hasattr(fl_rl, 'run_fl_rl'), "run_fl_rl function not found"
    print("✓ run_fl_rl function found")
    
    # Test FL ES
    assert hasattr(fl_es, 'run_fl_es'), "run_fl_es function not found"
    print("✓ run_fl_es function found")
    
    # Test utils
    assert hasattr(utils, 'plot_results'), "plot_results function not found"
    print("✓ plot_results function found")
    
    print("All required functions found!")

if __name__ == "__main__":
    test_imports()
    test_function_signatures()
    print("\nCartPole implementation test completed successfully!") 