sTDA Accelerated Davidson Algorithm
===================================

This is a Davidson code (Python3) ased on PySCF pilot codes, to solve excitation energies in TDA-TDDFT scheme.

You can specify different initial guess and preconditioner to compare their efficiency.

To run this code with command line:

$ python Davidson -c <xyzfile> -m <RHF/RKS/UHF/UKS> -f <xc_functiomal> -b <basis_set>
-g <grid_level> -i <sTDA_A/diag_A> -p <sTDA_A/diag_A> -t <tolerance> -n <No.states>
