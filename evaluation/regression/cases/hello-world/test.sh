#!/bin/bash
set -eo pipefail

# Function to display usage message
usage() {
    echo "Usage: $0 <directory_to_execute>"
}

# Check if the argument file is provided
if [ "$#" -ne 1 ]; then
    usage
    exit 1
fi

# Get the directory to execute from the command-line argument
directory_to_execute="$1"

# Define the expected output
expected_output="hello world"

# Find the .sh file to execute in the specified directory
file_to_execute=$(find "$directory_to_execute" -type f -name '*.sh')

# Check if the file was found
if [ -z "$file_to_execute" ]; then
    echo "Error: No .sh file found in the specified directory."
    exit 1
fi

# Run the shell script and capture its output
actual_output=$(sh "$file_to_execute")

# Compare the actual output to the expected output
if [ "$actual_output" == "$expected_output" ]; then
    # If they match, the test passed
    echo "Test passed: Output matches expected value."
    exit 0
else
    # If they don't match, the test failed
    echo "Test failed: Output does not match expected value."
    echo "Expected: $expected_output"
    echo "Actual: $actual_output"
    exit 1
fi
