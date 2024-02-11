# Semiempirical-accelerated Davidson Algorithm


This is a Davidson code (Python3) based on PySCF pilot codes, to calculate TDA/TDDFT excitation energies, static/dynamic polarizabilities and CPKS equations.

You can specify different initial guess and preconditioners to compare their efficiency.

To run this code, use command line:

```
python3 path_to_your_dir/Davidson/Davidson.py -x <molecule>.xyz -b def2-tzvp \
-f wb97x -n 5 -df True -TDA true -o 0 -M 60000 -v 5 -chk False \
-TV 1000000 40 -TO 1000000 40 \
-cl_aux_p True -cl_aux_d True -ex_aux_p False -ex_aux_d False \
-Cl_theta 0.6 -Ex_theta 0.6 \
-TDDFT_as True -ei 3 -max 12 -out rispd > rispd.out 2>&1
```

Note:

(1) sTDA implementation does not support UHF/UKS yet
(2) TDDFT_as meas TDDFT-ris

