sTDA Accelerated Davidson Algorithm
===================================

This is a Davidson code (Python3) ased on PySCF pilot codes, to solve excitation energies in TDA-TDDFT scheme.

You can specify different initial guess and preconditioner to compare their efficiency.

To run this code, use command line:

$ python Davidson -x <molecule.xyz> -m <RHF/RKS/UHF/UKS> -f <xc_functiomal> -b <basis_set>
-g <grid_level> -i <sTDA/Adiag> -p <sTDA/Adiag> -t <tolerance> -n <No.states> -df <True/False>

For example:

$ python Davidson -x methanol.xyz -m RKS -f b3lyp -b def2-SVP -g 9 -i sTDA -p sTDA -t 1e-5 -n 5 

And if you just run $ python Davidson, default setting is:

-c methanol.xyz -m RHF -f b3lyp -b def2-SVP -g 3 -i Adiag -p Adiag -t 1e-5 -n 4

Note:

(1) sTDA implementation does not support UHF/UKS yet.

(2) When calling for RHF/UHF and none hybride functional, it use wb97x's fitting parameters in sTDA.

(3) Density fittinf should be turned off if it starts with a converged density-fitting SCF checkpoint file
