#!/bin/bash

python ../Davidson.py -x *.xyz -b def2-tzvp -f pbe0 -n 5 -df True -TDA true -o 0 \
-v 3 -chk True -TDDFT_as True -ei 2 -max 10 
