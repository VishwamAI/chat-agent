#!/bin/bash

# This script parses the memory_usage.txt file and extracts the memory usage data.
# It filters out the necessary information and outputs the last few entries.

# Define the input file
input_file="/home/ubuntu/memory_usage.txt"

# Check if the input file exists
if [ ! -f "$input_file" ]; then
  echo "File not found: $input_file"
  exit 1
fi

# Extract memory usage data and output to a new file
grep 'KiB Mem' "$input_file" | tail -n 20 > /home/ubuntu/parsed_memory_usage.txt
