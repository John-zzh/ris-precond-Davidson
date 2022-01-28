


def Jacobi_preconditioner(residual, sub_eigenvalue, current_dic, full_guess,
                return_index = None, W_H = None, V_H = None, sub_A_H = None,
                DKapp = args.DKapp):
    '''(1-uu*)(A-Ω*I)t = -B
        (1-uu*)Kz = y
        Kz - uu*Kz = y
        Kz + αu = y
        z =  K^-1y - αK^-1u
       B is residual, we want to solve t (approximately)
       z approximates t
       z = (A-Ω*I)^(-1)*(-B) - α(A-Ω*I)^(-1)*u
        let K_inv_y = (A-Ω*I)^(-1)*(-B)
        and K_inv_u = (A-Ω*I)^(-1)*u
       z = K_inv_y - α*K_inv_u
       where α = [u*(A-Ω*I)^(-1)y]/[u*(A-Ω*I)^(-1)u]  (using uz = 0)
       first, solve (A-Ω*I)^(-1)y and (A-Ω*I)^(-1)u
    '''
    B = residual
    omega = sub_eigenvalue
    u = full_guess

    if DKapp == False:
        K_inv_y, NA_dic = sTDA_preconditioner(-B, omega)
        K_inv_u, NA_dic = sTDA_preconditioner(u, omega)
    else:
        K_inv_y = TDA_A_diag_preconditioner(
                                residual=-B, sub_eigenvalue=omega, hdiag=hdiag)
        K_inv_u = TDA_A_diag_preconditioner(
                                residual=u, sub_eigenvalue=omega, hdiag=hdiag)

    n = np.multiply(u, K_inv_y).sum(axis=0)
    d = np.multiply(u, K_inv_u).sum(axis=0)
    Alpha = n/d
    print('N in Jacobi =', np.average(n))
    print('D in Jacobi =', np.average(d))
    print('Alpha in Jacobi =', np.average(Alpha))

    z = K_inv_y -  Alpha*K_inv_u

    return z, current_dic

def on_the_fly_Hx(W, V, sub_A, x):
    def Qx(V, x):
        '''Qx = (1 - V*V.T)*x = x - V*V.T*x'''
        VX = np.dot(V.T,x)
        x -= np.dot(V,VX)
        return x
    '''on-the-fly compute H'x
       H′ ≡ W*V.T + V*W.T − V*a*V.T + Q*K*Q
       K approximates H, here K = sTDA_A
       H′ ≡ W*V.T + V*W.T − V*a*V.T + (1-V*V.T)sTDA_A(1-V*V.T)
       H′x ≡ a + b − c + d
    '''
    a = einsum('ij, jk, kl -> il', W, V.T, x)
    b = einsum('ij, jk, kl -> il', V, W.T, x)
    c = einsum('ij, jk, kl, lm -> im', V, sub_A, V.T, x)
    d = Qx(V, sTDA_mv(Qx(V, x)))
    Hx = a + b - c + d
    return Hx

def new_ES(full_guess, return_index, W_H, V_H, sub_A_H,
                        residual=None, sub_eigenvalue=None, current_dic=None):
    '''new eigenvalue solver, to diagonalize the H'
       the traditional Davidson to diagonalize the H' matrix
       W_H, V_H, sub_A_H are from the exact H
    '''
    new_ES_start = time.time()
    tol = args.eigensolver_tol
    max = 30

    k = args.nstates
    m = min([k+8, 2*k, A_size])

    V = np.zeros((A_size, max*k + m))
    W = np.zeros_like(V)

    '''sTDA as initial guess'''
    V = sTDA_eigen_solver(m, V)
    W[:,:m] = on_the_fly_Hx(W_H, V_H, sub_A_H, V[:, :m])

    for i in range(max):
        sub_A = np.dot(V[:,:m].T, W[:,:m])
        sub_eigenvalue, sub_eigenket = np.linalg.eigh(sub_A)
        residual = np.dot(W[:,:m], sub_eigenket[:,:k])
        residual -= np.dot(V[:,:m], sub_eigenket[:,:k] * sub_eigenvalue[:k])

        r_norms = np.linalg.norm(residual, axis=0).tolist()
        max_norm = np.max(r_norms)
        if max_norm < tol or i == (max-1):
            break
        index = [r_norms.index(i) for i in r_norms if i > tol]

        new_guess = TDA_A_diag_preconditioner(
                                          residual=residual[:,index],
                                    sub_eigenvalue=sub_eigenvalue[:k][index],
                                             hdiag=hdiag)
        V, new_m = mathlib.Gram_Schmidt_fill_holder(V, m, new_guess)
        W[:, m:new_m] = on_the_fly_Hx(W_H, V_H, sub_A_H, V[:, m:new_m])
        m = new_m

    full_guess = np.dot(V[:,:m], sub_eigenket[:,:k])

    new_ES_end = time.time()
    new_ES_cost = new_ES_end - new_ES_start
    print('H_app diagonalization done in',i,'steps; ','%.2f'%new_ES_cost, 's')
    print('threshold =', tol)
    return full_guess[:,return_index], current_dic
