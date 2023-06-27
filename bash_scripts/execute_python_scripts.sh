#!/bin/bash

# Array of Python scripts to execute
scripts=(
    "Kleinfeld_Network_Script.py"
    "Smith_Network_Script.py"
)

# Function to execute a script
execute_script() {
    python "$1"
}

# Run each script in the background
for script in "${scripts[@]}"; do
    execute_script "$script" &
done

# Wait for all background processes to finish
wait

