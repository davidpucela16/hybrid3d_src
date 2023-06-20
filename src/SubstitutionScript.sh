#!/bin/bash

# Array of search phrases
search_phrases=("script_path", "script_dir", "Output_Folder", "mat_path" , "output_dir_network", "sol_path")
# Array of replacement phrases
replacement_phrases=("script", "path_script", "path_output", "path_matrices" ,"path_output_network", "path_output_data")
 


# Check if the number of search phrases and replacement phrases match
if [ ${#search_phrases[@]} -ne ${#replacement_phrases[@]} ]; then
  echo "Error: The number of search phrases and replacement phrases does not match."
  exit 1
fi

# Execute substitution for each pair of search and replacement phrases
for (( i=0; i<${#search_phrases[@]}; i++ )); do
  search="${search_phrases[$i]}"
  replacement="${replacement_phrases[$i]}"
  command="grep -rlZ '$search' ./ | xargs -0 sed -i 's/$search/$replacement/g'"
  echo "Executing command: $command"
  eval $command
done
