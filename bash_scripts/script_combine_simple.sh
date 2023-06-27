#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <file1> <file2> [<file3> ...]"
    exit 1
fi

# Get the output file name
output_file="combined_full.am"

# Join the files in order
cat "$1" > "$output_file"
shift
while [ "$#" -gt 0 ]; do
    cat "$1" >> "$output_file"
    shift
done


