# -*- coding: utf-8 -*-

from arguments import args
from sTDA import sTDA_mv_lib
from TDDFT_as import TDDFT_as_lib


a = [args.sTDA, args.TDDFT_as]
b = [sTDA_mv_lib.sTDA(), TDDFT_as_lib.TDDFT_as()]

for i in range(len(a)):
    if a[i] == True:
        approx_TDA_mv, approx_TDDFT_mv, approx_spolar_mv = b[i].build()
