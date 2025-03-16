#!/usr/bin/env python3
"""
This script tests that nano_llm fails to import when transformers is missing.
It is specifically configured to work with the nano_llm module at /root/repo/NanoLLM.
"""

import sys
import os
import subprocess

# Hardcoded path to the nano_llm module
NANO_LLM_PATH = "/root/repo/NanoLLM"

def main():
    # Verify the module path exists
    if not os.path.exists(NANO_LLM_PATH):
        print(f"ERROR: Module path {NANO_LLM_PATH} does not exist")
        return False
    
    # Create a temporary Python script that tries to import nano_llm
    test_script = """
import sys
try:
    # Try to import nano_llm
    import nano_llm
    print("SUCCESS: nano_llm imported successfully")
    sys.exit(0)
except ImportError as e:
    error_message = str(e)
    if "No module named 'transformers'" in error_message:
        print("Failed to load nano_llm: No module named 'transformers'")
        sys.exit(1)
    else:
        print(f"Failed to load nano_llm with an unexpected error: {error_message}")
        sys.exit(2)
"""
    # Write the script to a temporary file in the current directory
    temp_file = os.path.join(os.getcwd(), "temp_import_test.py")
    with open(temp_file, "w") as f:
        f.write(test_script)
    
    try:
        # Run the script in a subprocess with PYTHONPATH set to include the module path
        env = os.environ.copy()
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{NANO_LLM_PATH}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = NANO_LLM_PATH
        
        print(f"Testing with module path: {NANO_LLM_PATH}")
        result = subprocess.run([sys.executable, temp_file], 
                               capture_output=True, text=True, env=env)
        
        # Check if the output contains the transformers error message
        if "No module named 'transformers'" in result.stdout and result.returncode == 1:
            print("TEST PASSED: nano_llm correctly fails when transformers is missing")
            return True
        else:
            print("TEST FAILED: Unexpected output or return code")
            print(f"Output: {result.stdout}")
            print(f"Return code: {result.returncode}")
            return False
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 