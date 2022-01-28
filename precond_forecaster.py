import os, sys
import numpy as np



class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def gen_metrics(beta, alpha, ab_initio_V, ab_initio_E,
                    full_sTDA_mv, sTDA_mv, AV, residual,
                                V_basis, W_basis, sub_A):
    V = ab_initio_V #ab-initio eigenkets
    lambda_matrix = np.diag(ab_initio_E)

    with HiddenPrints():
        '''sTDA V and E
           energies_se is in eV
        '''
        V_se, energies_se = sTDA_eigen_solver(k=args.nstates,
                                            tol=args.conv_tolerance,
                          matrix_vector_product=sTDA_mv)
    '''energy difference
    '''
    energy_diff = np.linalg.norm(energies_se - ab_initio_E*parameterlib.Hartree_to_eV)

    V_overlap = np.linalg.norm(np.dot(V_se.T, V))

    if args.AV == True:
        ab_initio_AV = AV
    else:
        ab_initio_AV = ab_initio_E * V

    ''' commutator norm
        |V.T[A, A^se]V|
       =[λ, V.T*A^se*V]
    '''
    VAseV = np.dot(V.T, full_sTDA_mv(V))
    commutator = mathlib.commutator(lambda_matrix, VAseV)
    VcAAV = np.linalg.norm(commutator)

    ''' 2nd promising!
        condition number
        |V.T * A^{se,-1}A * V|
        AV = λ * V
        A^{se,-1} AV = X => A^se X = AV
    '''
    with HiddenPrints():
        '''A^se X=W
        '''
        X = stapol_sTDDFT_initprec(Pr=-ab_initio_AV,
                                  tol=1e-6,
                matrix_vector_product=sTDA_mv)
    sub_condition = np.dot(V.T, X)
    k_VA_1AV = mathlib.cond_number(sub_condition)

    '''commutator basis V norm
        here, V_b is the Kryove space basis, W = AV
        |V_b.T [A, A^se] V_b|
       = V_b.T (A A^se - A^se A) V_b
       = V_b.T A A^se V_b - V_b.T A^se A V_b
       = W_b.T W^se - W^se.T W_b
    '''
    AseVb = full_sTDA_mv(V_basis)
    WbAseVb = np.dot(W_basis.T, AseVb)
    VbcAAVb = np.linalg.norm(WbAseVb - WbAseVb.T)

    ''' most promising!
        quality norm
        |(I - A^{se,-1}A) V |
        = V - X
    '''
    I_AAV = np.linalg.norm(V - X)

    metric_list = [
        beta,
        alpha,
        I_AAV,
        k_VA_1AV,
        VcAAV,
        VbcAAVb,
        energy_diff,
        V_overlap]

    return metric_list

def gen_I_AAV(ab_initio_E:list, ab_initio_V:np.array,
                            matrix_vector_product=sTDA_mv) -> float:
    with HiddenPrints():
        '''A^se X= AV
        '''
        AV = ab_initio_E * ab_initio_V
        X = stapol_sTDDFT_initprec(Pr=-AV, tol=1e-6,
                            matrix_vector_product=matrix_vector_product)

    I_AAV = float(np.linalg.norm(ab_initio_V - X))
    return I_AAV

def gen_forecaster(eta, *args):
    '''  |(I - A^{se,-1}A) V |  = V - X
        a is chemical hardness for each atom, in shape (N_atm,)
    '''
    ab_initio_E, ab_initio_V = args

    I_AAV = gen_I_AAV(ab_initio_E=ab_initio_E, ab_initio_V=ab_initio_V,
                                                matrix_vector_product=sTDA_mv)

    return I_AAV

def gen_as_forecaster(U:np.array, *args):
    U = U.tolist()
    '''  |(I - A^{se,-1}A) V |  = V - X
        U is index for each atom, 1d array (N_atm,)
    '''
    ab_initio_E, ab_initio_V = args
    TDDFT_as = TDDFT_as_lib.TDDFT_as(U_list=U_list)

    TDDFT_as.build()
    as_mv = TDDFT_as.TDA_as_mv

    I_AAV = gen_I_AAV(ab_initio_E=ab_initio_E, ab_initio_V=ab_initio_V,
                                                matrix_vector_product=as_mv)

    return I_AAV
