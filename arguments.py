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
    parser.add_argument('-chk', '--checkfile',      type=str2bool,  default=True, help='checkpoint filename (.chk)')
    parser.add_argument('-m', '--method',           type=str,   default='RKS', help='RHF RKS UHF UKS')
    parser.add_argument('-f', '--functional',       type=str,   default='pbe0',  help='xc functional')
    parser.add_argument('-b', '--basis_set',        type=str,   default='def2-svp',  help='basis set')
    parser.add_argument('-df', '--density_fit',     type=str2bool,  default=True,  help='density fitting turn on')
    parser.add_argument('-g', '--grid_level',       type=int,   default='3',   help='0-9, 9 is best')

    parser.add_argument('-n','--nstates',           type=int,   default = 5,      help='number of excited states')
    parser.add_argument('-pytd','--pytd',           type=str2bool,  default = False , help='whether to compare with PySCF TDDFT')

    parser.add_argument('-TDA','--TDA',             type=str2bool,  default = False, help='perform TDA')
    parser.add_argument('-TDDFT','--TDDFT',         type=str2bool,  default = False, help='perform TDDFT')
    parser.add_argument('-dpolar','--dpolar',       type=str2bool,  default = False, help='perform dynamic polarizability')
    parser.add_argument('-omega','--dpolar_omega',  type=float, default = [], nargs='+', help='dynamic polarizability with perurbation omega, a list')
    parser.add_argument('-spolar','--spolar',       type=str2bool,  default = False, help='perform static polarizability')
    parser.add_argument('-sTDA','--sTDA',           type=str2bool,  default = False, help='perform sTDA calculation')
    parser.add_argument('-sTDDFT','--sTDDFT',       type=str2bool,  default = False, help='perform sTDDFT calculation')
    parser.add_argument('-TDDFT_as','--TDDFT_as',   type=str2bool,  default = False, help='perform TDDFT_as calculation')

    parser.add_argument('-etatune','--etatune',     type=str2bool,  default = False, help='optimize accoriding to eta')
    parser.add_argument('-eta','--eta',             type=str2bool,  default = False, help='use the external eta set')
    parser.add_argument('-bounds','--bounds',       type=float,  default = 0.1, help='0.9-1.1')
    parser.add_argument('-step','--step',           type=float,  default = 1e-6, help='1e-2 - 1e-9')
    parser.add_argument('-ftol','--ftol',           type=float,  default = 1e-5, help='1e-2 - 1e-9')

    parser.add_argument('-Utune','--Utune',         type=str2bool,  default = False, help='optimize accoriding to U')
    parser.add_argument('-U','--U',                 type=str2bool,  default = False, help='optimize accoriding to U')

    parser.add_argument('-TV','--truncate_virtual', type=float, default = 40,    help='the threshold to truncate virtual orbitals, in eV')

    parser.add_argument('-o','--ip_options',        type=int,   default = [0], nargs='+', help='0-7')
    parser.add_argument('-t','--conv_tolerance',    type=float, default= 1e-5, help='residual norm Convergence threhsold')

    parser.add_argument('-it','--initial_TOL',      type=float, default= 1e-3, help='conv for the inital guess')
    parser.add_argument('-pt','--precond_TOL',      type=float, default= 1e-2, help='conv for TDA preconditioner')

    parser.add_argument('-ei','--extrainitial',     type=int,   default= 8,    help='number of extral TDA initial guess vectors, 0-8')
    parser.add_argument('-max','--max',             type=int,   default= 35,   help='max iterations')

    parser.add_argument('-M','--memory',            type=int,   default= 4000, help='max_memory')
    parser.add_argument('-v','--verbose',           type=int,   default= 3,    help='mol.verbose = 3,4,5')

    parser.add_argument('-beta','--beta',           type=float, default= None,  help='beta = 4.00')
    parser.add_argument('-alpha','--alpha',         type=float, default= None,  help='alpha = 0.83')

    parser.add_argument('-tuning','--tuning',       type=str2bool, default= False,    help='turn on on-the-fly tuning')

    parser.add_argument('-beta_list','--beta_list',   type=float, default= [],    nargs='+', help='8 7 6 5 4 3 2')
    parser.add_argument('-alpha_list','--alpha_list', type=float, default= [],    nargs='+', help='8 7 6 5 4 3 2 1.8 1 0.8')

    args = parser.parse_args()
    if args.dpolar == True and args.dpolar_omega == []:
        raise ValueError('External Perturbation ω cannot be None')
    return args

args = gen_args()
