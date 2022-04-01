import numpy as np
from scipy.linalg import solve_triangular
from matern_52 import matern_52
from scipy.linalg import cholesky
from numpy.random import standard_normal

def gp_pred(y1, L1, dx2, dx12, sig, l, delta):

    if np.isscalar(l):
        L1_div_y1 = solve_triangular(L1,y1,lower=True)
        Q_div_y1 = solve_triangular(L1.T,L1_div_y1)
        Q_x1_x2 = matern_52(dx12,sig,l)
        y2_mu = Q_x1_x2.T @ Q_div_y1
        v_pred = solve_triangular(L1,Q_x1_x2,lower=True)
        cov_y2 = matern_52(dx2,sig,l) - v_pred.T @ v_pred + np.identity(dx2.shape[0])*delta
        y2 = y2_mu + cholesky(cov_y2,lower=True) @ standard_normal(y2_mu.size)
    else:
        nit = l.shape[0]
        n2 = dx12.shape[1]
        Q_x1_x2 = matern_52(dx12,sig,l)
        Q_x2 = matern_52(dx2,sig,l)
        y2 = np.zeros((nit,n2))
        for i in range(nit):
            L1_div_y1 = solve_triangular(L1[i,:,:],y1[i,:],lower=True)
            Q_div_y1 = solve_triangular(L1[i,:,:].T,L1_div_y1)
            y2_mu = Q_x1_x2[i,:,:].T @ Q_div_y1
            v_pred = solve_triangular(L1[i,:,:],Q_x1_x2[i,:,:],lower=True)
            cov_y2 = Q_x2[i,:,:] - v_pred.T @ v_pred + np.identity(dx2.shape[0])*delta
            y2[i,:] = y2_mu + cholesky(cov_y2,lower=True) @ standard_normal(y2_mu.size)
    return y2
