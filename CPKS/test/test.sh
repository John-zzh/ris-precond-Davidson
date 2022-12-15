#!/bin/bash

# python ../Davidson.py -x *.xyz -b def2-svp -f pbe0 -n 5 -df True -TDDFT true -o 0 \
# -v 4 -chk true -TDDFT_as True -ei 2 -max 10 \
# -dscf true \
# -coulomb_ex none \
# -cl_aux_p true -ex_aux_p False \
# -Uc 1 -Ue 1 \
# -TV 1000000 40 -TO 1000000 40 \
# -TDDFT_as_profile true



python ../../Davidson.py -x *.xyz -b def2-svp -f pbe0 -n 5 -df True \
-v 3 -chk false -TDDFT_as True \
-dscf false \
-cl_aux_p True -ex_aux_p False \
-Uc 4 -Ue 4 \
-TV 1000000 1000000 -TO 1000000 1000000 \
-TDDFT_as_profile true \
-CPKS true -o 1 -t 1e-8