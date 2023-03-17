# -*- coding: utf-8 -*-

import argparse

def str2bool(str):
    if str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def gen_args():
    parser = argparse.ArgumentParser(description='Davidson')
    parser.add_argument('-x', '--xyzfile',          type=str,   default='methanol.xyz',  help='xyz filename (molecule.xyz)')
    parser.add_argument('-chk', '--checkfile',      type=str2bool,  default=True, help='use checkpoint file')
    parser.add_argument('-dscf', '--dscf',      type=str2bool,  default=False, help='0 SCF iterations')
    parser.add_argument('-m', '--method',           type=str,   default='RKS', help='RHF RKS UHF UKS')
    parser.add_argument('-f', '--functional',       type=str,   default=None,  help='xc functional')
    parser.add_argument('-b', '--basis_set',        type=str,   default='def2-svp',  help='basis set')
    parser.add_argument('-df', '--density_fit',     type=str2bool,  default=True,  help='density fitting turn on')
    parser.add_argument('-g', '--grid_level',       type=int,   default='3',   help='0-9, 9 is best')
    parser.add_argument('-st', '--scf_tolerence',   type=float,  default=1e-10,   help='SCF convergence tolerence')
    parser.add_argument('-c', '--charge',           type=int,   default=0,   help='molecular net charge')
    parser.add_argument('-sg', '--singular',        type=str2bool,   default=False,   help='remove singularirty')


    parser.add_argument('-n','--nstates',           type=int,   default = 5,      help='number of excited states')
    parser.add_argument('-pytd','--pytd',           type=str2bool,  default = False , help='whether to compare with PySCF TDDFT')

    parser.add_argument('-TDA','--TDA',             type=str2bool,  default = False, help='perform TDA')
    parser.add_argument('-TDDFT','--TDDFT',         type=str2bool,  default = False, help='perform TDDFT')
    parser.add_argument('-dpolar','--dpolar',       type=str2bool,  default = False, help='perform dynamic polarizability')
    parser.add_argument('-omega','--dpolar_omega',  type=float, default = [], nargs='+', help='dynamic polarizability with perurbation omega, a list')
    parser.add_argument('-spolar','--spolar',       type=str2bool,  default = False, help='perform static polarizability')
    parser.add_argument('-CPKS','--CPKS',           type=str2bool,  default = False, help='perform CPKS')
    parser.add_argument('-sTDA','--sTDA',           type=str2bool,  default = False, help='perform sTDA calculation')
    parser.add_argument('-sTDDFT','--sTDDFT',       type=str2bool,  default = False, help='perform sTDDFT calculation')
    parser.add_argument('-TDDFT_as','--TDDFT_as',   type=str2bool,  default = False, help='perform TDDFT_as preconditioner')

    parser.add_argument('-cpks_test','--cpks_test',  type=str2bool,  default = False, help='compare to pyscf CPKS z1 solution')

    parser.add_argument('-TDA_as_profile','--TDA_as_profile',   type=str2bool,  default = False, help='perform TDA_as enery calculation')
    parser.add_argument('-TDDFT_as_profile','--TDDFT_as_profile',   type=str2bool,  default = False, help='perform TDDFT_as enery calculation')
    parser.add_argument('-spolar_as_profile','--spolar_as_profile',   type=str2bool,  default = False, help='perform spolar_as calculation')
    parser.add_argument('-dpolar_as_profile','--dpolar_as_profile',   type=str2bool,  default = False, help='perform dpolar_as calculation')

    parser.add_argument('-test','--test',   type=str2bool,  default = True, help='check methanol pbe0 energy')

    parser.add_argument('-spectra','--spectra',     type=str2bool,  default = False, help='plot excitaition spectra peaks')

    # parser.add_argument('-etatune','--etatune',     type=str2bool,  default = False, help='optimize accoriding to eta')
    parser.add_argument('-eta','--eta',             type=str2bool,  default = False, help='use the external eta set')
    # parser.add_argument('-bounds','--bounds',       type=float,  default = 0.1, help='0.9-1.1')
    # parser.add_argument('-step','--step',           type=float,  default = 1e-4, help='1e-2 - 1e-9')
    # parser.add_argument('-ftol','--ftol',           type=float,  default = 1e-3, help='1e-2 - 1e-9')

    # parser.add_argument('-Uread','--Uread',         type=str2bool,  default = False, help='read Uk value from txt file')
    # parser.add_argument('-coulomb_ex','--coulomb_ex', type=str,  default = 'coulomb', help='coulomb & exchange & all & none')

    parser.add_argument('-cl_aux_p','--coulomb_aux_add_p',  type=str2bool,  default = False, help='s or sp')
    parser.add_argument('-cl_aux_d','--coulomb_aux_add_d',  type=str2bool,  default = False, help='turn on spd')
    parser.add_argument('-ex_aux_p','--exchange_aux_add_p', type=str2bool,  default = False, help='s or sp')

    parser.add_argument('-w','--woodbury',          type=str2bool,  default = False, help='only keep diagonal elements in (ij|ab)')
    parser.add_argument('-FK','--full_fitting',          type=str2bool,  default = False, help='full fitting basis for HFX')
    # parser.add_argument('-Uconst','--Uconst',       type=float,  default = 0.0, help='use a constant 0.123 as s and p orbital exponential')
    parser.add_argument('-Uc','--coulomb_U',         type=float,  default = 1.0, help='use Uc/R^2 as orbital exponent for coulomb term')
    parser.add_argument('-Ue','--exchange_U',        type=float,  default = 1.0, help='use Ue/R^2 as orbital exponent for exchange term')
    parser.add_argument('-mix_c','--mix_c',         type=float,  default = 1.0, help='use Ue/R^2 as orbital exponent for exchange term')


    parser.add_argument('-Uk_tune','--Uk_tune',     type=str2bool,  default = False, help='tune the k parameter of k/R**0.5')

    parser.add_argument('-TV','--truncate_virtual',  type=float, default = [100000,100000], nargs='+', help='the threshold to truncate virtual orbitals, in eV or %')
    parser.add_argument('-TO','--truncate_occupied', type=float, default = [100000,100000], nargs='+', help='the threshold to truncate occupied orbitals, in eV or %')
    # parser.add_argument('-Tex','--truncate_exchange_only', type=str2bool, default = True,    help='only truncate the exchange integrals')
    parser.add_argument('-GS','--GS_double',        type=str2bool, default = False, help='double filter in GS orthonormalization')

    parser.add_argument('-o','--ip_options',        type=int,   default = [0], nargs='+', help='0-7')
    parser.add_argument('-t','--conv_tolerance',    type=float, default= 1e-5, help='residual norm Convergence threhsold')

    parser.add_argument('-it','--initial_TOL',      type=float, default= 1e-3, help='conv for the inital guess')
    parser.add_argument('-pt','--precond_TOL',      type=float, default= 1e-2, help='conv for TDA preconditioner')
    parser.add_argument('-jacobi','--jacobi',       type=str2bool, default= False, help='turn on jacobi preconditioner')
    parser.add_argument('-projector','--projector', type=str2bool, default= False, help='turn on projector preconditioner')
    parser.add_argument('-approx_p','--approx_p',   type=str2bool, default= True, help='turn on semiempirical appriximation in projector&jacobi preconditioner')

    parser.add_argument('-ei','--extrainitial',     type=int,   default= 8,    help='number of extral TDA initial guess vectors, 0-8')
    parser.add_argument('-max','--max',             type=int,   default= 35,   help='max iterations')

    parser.add_argument('-M','--memory',            type=int,   default= 4000, help='max_memory')
    parser.add_argument('-v','--verbose',           type=int,   default= 3,    help='mol.verbose = 3,4,5')

    parser.add_argument('-einsum','--einsum',       type=str,   default= 'pyscf',    help='pyscf, opt, parallel')

    parser.add_argument('-beta','--beta',           type=float, default= None,  help='beta = 4.00')
    parser.add_argument('-alpha','--alpha',         type=float, default= None,  help='alpha = 0.83')
    #
    # parser.add_argument('-tuning','--tuning',       type=str2bool, default= False,    help='turn on on-the-fly tuning')
    #
    # parser.add_argument('-beta_list','--beta_list',   type=float, default= [],    nargs='+', help='8 7 6 5 4 3 2')
    # parser.add_argument('-alpha_list','--alpha_list', type=float, default= [],    nargs='+', help='8 7 6 5 4 3 2 1.8 1 0.8')

    args = parser.parse_args()
    if args.dpolar == True and args.dpolar_omega == []:
        raise ValueError('External Perturbation ω cannot be None')
    return args

args = gen_args()

if args.CPKS:
    args.nstates = 1
elif args.dpolar or args.spolar:
    args.nstates = 3
