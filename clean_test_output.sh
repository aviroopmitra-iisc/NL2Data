#!/bin/bash
# Script to forcefully delete all contents inside test_output directory (no confirmation)

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_OUTPUT_DIR="${SCRIPT_DIR}/test_output"

# Check if test_output directory exists
if [ ! -d "$TEST_OUTPUT_DIR" ]; then
    echo "Error: test_output directory does not exist: $TEST_OUTPUT_DIR"
    exit 1
fi

# Count files/directories before deletion
COUNT=$(find "$TEST_OUTPUT_DIR" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)

if [ "$COUNT" -eq 0 ]; then
    echo "test_output directory is already empty."
    exit 0
fi

echo "Deleting $COUNT item(s) from: $TEST_OUTPUT_DIR"

# Delete all contents (including hidden files)
rm -rf "${TEST_OUTPUT_DIR}"/* "${TEST_OUTPUT_DIR}"/.[!.]* "${TEST_OUTPUT_DIR}"/..?* 2>/dev/null

# Check if deletion was successful
if [ $? -eq 0 ]; then
    echo "Successfully deleted all contents in test_output directory."
else
    echo "Error: Some files may not have been deleted."
    exit 1
fi

