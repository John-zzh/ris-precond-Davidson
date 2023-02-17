

python ../../Davidson.py -x methanol.xyz -b def2-svp -f pbe0 -df True \
-v 3 -chk false -TDDFT_as True \
-cl_aux_p true -ex_aux_p false \
-Uc 1 -Ue 1 \
-TV 1000000 1000000 -TO 1000000 1000000 \
-spolar_as_profile true \
-spolar true -o 0 -t 1e-5 \
-w true -FK true
