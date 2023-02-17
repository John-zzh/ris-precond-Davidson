#!/bin/bash

python ../../Davidson.py -x *.xyz -b def2-svp -f pbe0 -n 5 -ei 2 -df True \
-v 3 -chk false -TDDFT_as True \
-cl_aux_p true -ex_aux_p false \
-Uc 1 -Ue 1 \
-TV 1000000 1000000 -TO 1000000 1000000 \
-TDA true  -TDA_as_profile true \
-w true -FK true -mix_c 0.2
