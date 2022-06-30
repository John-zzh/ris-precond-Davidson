# -*- coding: utf-8 -*-

from arguments import args

if args.sTDA == True:
    from sTDA.sTDA_mv_lib import sTDA as approx

if args.TDDFT_as == True:
    from TDDFT_as.TDDFT_as_lib import TDDFT_as as approx


approx_mv = approx()

approx_TDA_mv, approx_TDDFT_mv, approx_spolar_mv = approx_mv.build()
