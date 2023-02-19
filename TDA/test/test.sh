#!/bin/bash

python ../../Davidson.py -x *.xyz -b def2-svp -f pbe0 -n 5 -ei 2 -df True \
-v 3 -chk false -TDDFT_as True \
-cl_aux_p false -ex_aux_p false \
-Uc 0.2 -Ue 0.2 \
-TV 1000000 1000000 -TO 1000000 1000000 \
-TDDFT true  -TDDFT_as_profile true \
-w false -FK false -mix_c 0
