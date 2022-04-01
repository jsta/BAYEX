import numpy as np

def matern_52(r,s,l):
    if np.isscalar(s):
        q = s*s*(1 + np.sqrt(5)/l*r + 5/(3*l*l)*(r * r)) * np.exp(-np.sqrt(5)/l*r)
    else:
        n1,n2 = r.shape
        n3 = s.shape[0]
        s = np.repeat(s,n1*n2,axis=1)
        l = np.repeat(l,n1*n2,axis=1)
        r = r.flatten('F')[:,None].T
        r = np.repeat(r,n3,axis=0)
        q = s*s*(1 + np.sqrt(5)/l*r + 5/(3*l*l)*(r * r)) * np.exp(-np.sqrt(5)/l*r);
        q = np.reshape(q,(n3,n1,n2),order='F')
    return q
