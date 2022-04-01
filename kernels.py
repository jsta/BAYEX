import numpy as np

def kernels(r,tau):

    if np.isscalar(tau):
        w = np.exp(-0.6931/(tau*tau)*r*r);
        wn = np.ones((1,w.shape[0])) @ w
        w = w / wn
    else:
        tau = tau.T
        nl,npr = r.shape
        nit = tau.shape[1]
        taui = np.repeat(tau * tau,nl*npr,axis=0)
        ri = np.repeat((r.flatten('F') * r.flatten('F'))[:,None],nit,axis=1)
        w = np.exp(-0.6931/taui * ri)
        w = np.reshape(w,(nl,npr*nit),order='F')
        wn = np.ones((1,nl)) @ w
        w = w / wn
        w = np.reshape(w,(nl,npr,nit),order='F')
    return w
