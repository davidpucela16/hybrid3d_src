#!/bin/bash

#!/bin/bash

# Arguments for the Python script
arg1="$1"
arg2="$2"
arg3="$3"

sed -i "s/cells_3D=/#cells_3D=/" Kleinfeld_Network_Script.py                                                 
sed -i "s/n=/#n=/" Kleinfeld_Network_Script.py
sed -i "s/#cells_3D, n/cells_3D, n/" Kleinfeld_Network_Script.py

sed -i "s/#string_value =/string_value =/" Kleinfeld_Network_Script.py

# Execute the Python script with arguments
python Kleinfeld_Network_Script.py "$arg1" "$arg2" "$arg3"

# Wait for all background processes to finish
wait

