#!/bin/bash

# python ../../Davidson.py -x *.xyz -b def2-svp -f pbe0 -n 5 -ei 2 -df True \
# -v 3 -chk false -TDDFT_as True \
# -cl_aux_p true -cl_aux_d true  -ex_aux_p false \
# -Uc 0.2 -Ue 0.2 \
# -TV 1000000 1000000 -TO 1000000 1000000 \
# -TDA true  -TDA_as_profile true \
# -w false -FK false -mix_c 0


python3 ../../Davidson.py -x benzene.xyz -b def2-svp -f b3lyp -df True \
-v 5 -chk true -TDDFT_as true -sTDA false \
-cl_aux_p false -cl_aux_d false  -ex_aux_p false \
-Cl_theta 0.2 -Ex_theta 0.2 \
-TV 1000000 1000000 -TO 1000000 1000000 \
-o 0 -TDDFT true  -TDDFT_as_profile true -FK true


# \
# # -w false -FK false -mix_c 0 -o 0 -jacobi false -projector false -approx_p false
# DFT-ris 0.2   SCF energy -150.447160470415 E1 = -260.5206264086395   Ecoul = 87.93938411168052  Exc = -18.211294055697262 HOMO = -1.33270328162688   LUMO = -0.646715120139003
# DFT-ris 1.0   SCF energy -124.902876337702 E1 = -239.3809312710022   Ecoul = 88.90366973836788  Exc = -14.770990687309808 HOMO = -0.174775326058641  LUMO = 0.0364455356626264
# DFT-risp1.0   SCF energy -124.930932931904 E1 = -238.9760025273503   Ecoul = 88.59848263799817  Exc = -14.898788924793214 HOMO = -0.221648023701037  LUMO = 0.0296674288172779
# DFT-rijk      SCF energy -115.635128345605 E1 = -237.11276816541664  Ecoul = 96.49347340047204  Exc = -15.361209462902531 HOMO = -0.287262196174836  LUMO = 0.0350072862401992
#
