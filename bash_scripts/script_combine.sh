cat ../../Synthetic_ROIs_300x300x300/Rea1_synthetic_x.Smt.SptGraph.am Edge_concentration.txt > combined.am
cat combined.am Entry_Exit.txt > combined_full.am
rm combined.am

#!/bin/bash

# Input file
input_file="combined_full.am"

# Temporary file for storing intermediate changes
temp_file="temp.txt"

# Find the line number of the first occurrence of "@7"
line_number=$(grep -n "@7" "$input_file" | head -n 1 | cut -d ":" -f 1)

# Copy the lines before the insertion point to the temporary file
head -n "$line_number" "$input_file" > "$temp_file"

# Append the desired lines to the temporary file
echo "EDGE { float Concentration } @8" >> "$temp_file"
echo "VERTEX { int label } @9" >> "$temp_file"

# Copy the remaining lines after the insertion point to the temporary file
tail -n +"$((line_number+1))" "$input_file" >> "$temp_file"

# Overwrite the input file with the changes
mv "$temp_file" "$input_file"

echo "Insertion complete!"

scp ./combined_full.am pdavid@brainmicrovisu:/mnt/bighome/pdavid/combined_full.am
