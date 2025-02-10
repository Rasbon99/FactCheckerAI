import subprocess
from datetime import datetime

def run_tests():
    # Define the paths to the test files
    test_files = [
        "GraphRAG/Unit_tests.py",
        "GraphRAG/Integration_tests.py",
        "UI_Tests.py"
        ]
    
    # Define the output file
    output_file = "test_output.txt"
    
    # Open the output file in append mode
    with open(output_file, "a") as f:
        # Get the current date and time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Write the current date and time to the file
        f.write(f"\n\n--- Test Execution at {current_time} ---\n")
        
        # Run each test file and redirect the output to the file
        for test_file in test_files:
            result = subprocess.run(["python", test_file], capture_output=True, text=True)
            f.write(f"\nOutput of {test_file}:\n")
            f.write(result.stdout)
            f.write(result.stderr)

if __name__ == "__main__":
    run_tests()