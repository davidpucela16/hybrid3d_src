#!/bin/bash

# Read the value of num_scripts from Smith_Network_Script.py

script_orig=$1
num_scripts=$(grep -oP 'num_processes=\K\d+' $script_orig)

sed -i 's/simple_plotting=True/simple_plotting=False/' $script_orig

python "$script_orig"
wait $!

script_name="${script_orig%.py}"

cp $script_orig rec_0_backup.py

back_up="rec_0_backup.py"

sed -i 's/Computation_bool=True/Computation_bool=False/' $back_up

line_number=$(grep -n "^name_script" "$back_up" | cut -d ":" -f 1)

# Construct the variable assignment line
variable_assignment="name_script='${script_name}'"

# Insert the variable assignment line below the identified line
sed -i "${line_number}a${variable_assignment}" "$back_up"



# Create an array to store the background process IDs
declare -a pids

sed -i 's/Computation_bool=True/Computation_bool=False/' rec_0_backup.py
sed -i 's/rec_bool=False/rec_bool=True/' rec_0_backup.py

for ((x=0; x<num_scripts; x++)); do
	script="rec_3D_$x.py"
	cp rec_0_backup.py "$script"	
	sed -i "s/process=0/process=$x/" "$script"
	sed -i "s/CheckLocalConservativenessFlowRate(/#CheckLocalCons/" "$script"
	sed -i 's/rec_bool=False/rec_bool=True/' "$script"
			    
	python "$script" &  # Execute the script in the background
	pids[$x]=$!  # Store the process ID
done



# Wait for all background processes to finish
for pid in ${pids[@]}; do
    wait $pid
done

# Remove the backup script
rm rec_0_backup.py

for ((x=0; x<num_scripts; x++)); do
    script="rec_3D_$x.py"
    rm "$script"
done
