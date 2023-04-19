#!/bin/bash

# python ../../Davidson.py -x *.xyz -b def2-svp -f pbe0 -n 5 -ei 2 -df True \
# -v 3 -chk false -TDDFT_as True \
# -cl_aux_p true -cl_aux_d true  -ex_aux_p false \
# -Uc 0.2 -Ue 0.2 \
# -TV 1000000 1000000 -TO 1000000 1000000 \
# -TDA true  -TDA_as_profile true \
# -w false -FK false -mix_c 0


python ../../Davidson.py -x methanol.xyz -b def2-svp -f pbe0 -n 5 -ei 2 -df True -dscf true \
-v 3 -chk false -TDDFT_as true -sTDA false -it 1e-6 -pt 1e-5 \
-cl_aux_p true -cl_aux_d true  -ex_aux_p false \
-Uc 1 -Ue 1 \
-TV 1000000 1000000 -TO 1000000 1000000 \
-TDA true  -TDA_as_profile true -jacobi false -projector false -approx_p false \
-o 0 -FK true -fxc_on True

# \
# -w false -FK false -mix_c 0 -o 0 -jacobi false -projector false -approx_p false
