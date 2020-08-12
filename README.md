sTDA Accelerated Davidson Algorithm
===================================

This is a Davidson code (Python3) ased on PySCF pilot codes, to solve excitation energies in TDA-TDDFT scheme.

You can specify different initial guess and preconditioner to compare their efficiency.

To run this code, use command line:

$ python Davidson -c <molecule.xyz> -m <RHF/RKS/UHF/UKS> -f <xc_functiomal> -b <basis_set>
-g <grid_level> -i <sTDA_A/diag_A> -p <sTDA_A/diag_A> -t <tolerance> -n <No.states>

For example:

$ python Davidson -c methanol.xyz -m RKS -f b3lyp -b def2-SVP -g 9 -i sTDA_A -p sTDA_A -t 1e-5 -n 5

And if you just run $ python Davidson, default setting is:

-c methanol.xyz -m RHF -f b3lyp -b def2-SVP -g 3 -i diag_A -p diag_A -t 1e-5 -n 4
