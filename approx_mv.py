# -*- coding: utf-8 -*-

from arguments import args
from sTDA import sTDA_mv_lib
from TDDFT_as import TDDFT_as_lib



if args.sTDA == True:
    sTDA = sTDA_mv_lib.sTDA()
    sTDA_mv, full_sTDA_mv, TDDFT_mv, sTDDFT_spolar_mv = sTDA.build()

    (approx_TDA_mv,
    approx_TDDFT_mv,
    approx_spolar_mv) = sTDA_mv, TDDFT_mv, sTDDFT_spolar_mv

elif args.TDDFT_as == True:
    TDDFT_as = TDDFT_as_lib.TDDFT_as()
    TDA_as_mv, TDDFT_as_mv, TDDFT_as_spolar_mv = TDDFT_as.build()

    (approx_TDA_mv,
    approx_TDDFT_mv,
    approx_spolar_mv) = TDA_as_mv, TDDFT_as_mv, TDDFT_as_spolar_mv
