import numpy as np
from scipy.io import savemat,loadmat
from scipy.stats import genextreme as gev
from scipy.stats import gumbel_r as gumbel
from numpy.linalg import norm
from kernels import kernels
from matern_52 import matern_52
from scipy.linalg import cholesky
from gp_pred import gp_pred
from scipy.linalg import solve
from numpy.random import standard_normal

path2data = '/path/to/stan/model/output/' # path to the Stan model output (path to the file from 'saveEstimates.py')
pathOut = '/path/to/output/' # path where predictions will be saved
inFile = 'max_stable_estimates.mat' # name of the file that contains the Stan model output (from 'saveEstimates.py')
outFile = 'max_stable_predictions.mat' # name of the file where to save the predictions
data = loadmat(path2data + inFile)

nngp = data["nngp"][0][0] # nearest neighbor Gaussian process?
nnknot = data["nnknot"][0][0] # nearest neighbor residual process?
xi_spatial = data["xi_spatial"][0][0] # shape parameter spatial?
trend_linear = data["trend_linear"][0][0] # model the GEV location parameter as a linear trend?
is_gumbel = data["is_gumbel"][0][0] # set the value of the GEV shape parameter equal to zero (are we using a Gumbel distribution)?
delta = 1e-12 # used to deal with numerical non-positive definiteness

npr = data["loni"].shape[0]
nx,no = (data["y"].T).shape
nit = data["alpha"].shape[0]
if nnknot == 0:
    nl = data["lonk"].shape[0]
else:
    nl = data["Mk"][0][0]

# predict missing annual maxima at data sites
knan = np.argwhere(np.isnan(data["y"].flatten()))[:,0]
mu_star = np.reshape(data["mu_star"],(nit,nx*no),order='F')
sigma_star = np.reshape(data["sigma_star"],(nit,nx*no),order='F')
if is_gumbel == 0:
    if xi_spatial == 0:
        xi_star = np.repeat(data["xi_star"],knan.shape[0],axis=1)
    elif xi_spatial == 1:
        xi_star = np.repeat(data["xi_star"].flatten('F')[:,None],no,axis=1)
        xi_star = np.reshape(xi_star,(nit,nx*no),order='F')
        xi_star = xi_star[:,knan]
    ynan = gev.rvs(-xi_star,mu_star[:,knan],sigma_star[:,knan])
else:
    ynan = gumbel.rvs(mu_star[:,knan],sigma_star[:,knan])
yi = np.zeros((nit,nx*no)) * np.NaN
yi[:,knan] = ynan
yi = np.reshape(yi,(nit,nx,no),order='F')

# compute distance
if nngp == 0:
    lo = np.repeat(data["lon"],nx,axis=1)
    la = np.repeat(data["lat"],nx,axis=1)
    dx = norm(np.array((lo,la)) - np.array((lo.T,la.T)),axis=0,ord=2)/10.
    
    loi = np.repeat(data["loni"],npr,axis=1)
    lai = np.repeat(data["lati"],npr,axis=1)
    dx2 = norm(np.array((loi,lai)) - np.array((loi.T,lai.T)),axis=0,ord=2)/10.
    
    lo = np.repeat(data["lon"],npr,axis=1)
    la = np.repeat(data["lat"],npr,axis=1)
    loi = np.repeat(data["loni"].T,nx,axis=0)
    lai = np.repeat(data["lati"].T,nx,axis=0)
    dx12 = norm(np.array((lo,la)) - np.array((loi,lai)),axis=0,ord=2)/10.
    lo = None
    la = None
    loi = None
    lai = None
elif nngp == 1:
    M = data["M"][0][0]
    indM = np.zeros((M,npr),dtype=np.intp)
    dx12 = np.zeros((M,npr))
    dx = np.zeros((M,M,npr))
    for i in range(npr):
        di = norm(np.array((data["lon"][:,0],data["lat"][:,0])) - np.array((data["loni"][i,:],data["lati"][i,:])),axis=0,ord=2)/10.
        inds = np.argsort(di)[0:M]
        dx12[:,i] = di[inds]
        indM[:,i] = inds
        
        lo = np.repeat(data["lon"][inds,:],M,axis=1)
        la = np.repeat(data["lat"][inds,:],M,axis=1)
        dx[:,:,i] = norm(np.array((lo,la)) - np.array((lo.T,la.T)),axis=0,ord=2)/10.
        lo = None
        la = None
        
if nnknot == 0:
    lo = np.repeat(data["lonk"],npr,axis=1)
    la = np.repeat(data["latk"],npr,axis=1)
    loi = np.repeat(data["loni"].T,nl,axis=0)
    lai = np.repeat(data["lati"].T,nl,axis=0)
    dl = norm(np.array((lo,la)) - np.array((loi,lai)),axis=0,ord=2)/10.
    lo = None
    la = None
    loi = None
    lai = None
elif nnknot == 1:
    Mk = data["Mk"][0][0]
    indMk = np.zeros((Mk,npr),dtype=np.intp)
    dl = np.zeros((Mk,npr))
    for i in range(npr):
        di = norm(np.array((data["lonk"][:,0],data["latk"][:,0])) - np.array((data["loni"][i,:],data["lati"][i,:])),axis=0,ord=2)/10.
        inds = np.argsort(di)[0:Mk]
        dl[:,i] = di[inds]
        indMk[:,i] = inds


wl = kernels(dl,data["tau"])
alphai = np.repeat(data["alpha"].T,nl*npr,axis=0)
alphai = np.reshape(alphai,(nl,npr,nit),order='F')
wl = np.power(wl,1/alphai)
A_wl = np.zeros((nit,npr,no))
if nnknot == 0:
    for i in range(nit):
        A_wl[i,:,:] = (data["A"][i,:,:] @ wl[:,:,i]).T
elif nnknot == 1:
    for j in range(npr):
        for i in range(nit):
            Ai = data["A"][i,:,:]
            A_wl[i,j,:] = Ai[:,indMk[:,j]] @ wl[:,j,i]
A_wl = np.reshape(A_wl,(nit*npr,no),order='F')

scale_b00 = 100. if trend_linear else 500.
if nngp == 0:
    # compute Cholesky of kernel functions
    L_b10 = np.zeros((nit,nx,nx))
    L_b00 = np.zeros((nit,nx,nx))
    L_s = np.zeros((nit,nx,nx))
    Ddelta = np.identity(nx)*delta
    Q_b10 = matern_52(dx,data["s_b10"],data["l_b10"])
    Q_b00 = matern_52(dx,data["s_b00"]/scale_b00,data["l_b00"])
    Q_s = matern_52(dx,data["s_sigma"],data["l_sigma"])
    if not trend_linear:
        L_b0 = np.zeros((nit,nx,nx))
        Q_b0 = matern_52(dx,data["s_b0"]/1000,data["l_b0"])
    if xi_spatial == 1 and is_gumbel == 0:
        L_xi = np.zeros((nit,nx,nx))
        Q_xi = matern_52(dx,data["s_xi"],data["l_xi"])
    for i in range(nit):
        L_b10[i,:,:] = cholesky(Q_b10[i,:,:] + Ddelta,lower=True)
        L_b00[i,:,:] = cholesky(Q_b00[i,:,:] + Ddelta,lower=True)
        L_s[i,:,:] = cholesky(Q_s[i,:,:] + Ddelta,lower=True)
        if not trend_linear:
            L_b0[i,:,:] = cholesky(Q_b0[i,:,:] + Ddelta,lower=True)
        if xi_spatial == 1 and is_gumbel == 0:
            L_xi[i,:,:] = cholesky(Q_xi[i,:,:] + Ddelta,lower=True)
    #make predictions
    b1_pred = np.zeros((nit,npr,no))
    b10_pred = (data["xpred"] @ data["beta_b1"].T).T + gp_pred(data["b10"] - (data["xp"] @ data["beta_b1"].T).T,L_b10,dx2,dx12,\
    data["s_b10"],data["l_b10"], delta)
    b00_pred = gp_pred(data["b00"],L_b00,dx2,dx12,data["s_b00"]/scale_b00,data["l_b00"],delta)
    if not trend_linear:
        b0_pred = np.zeros((nit,npr,no))
    for j in range(no):
        if trend_linear:
            b1_pred[:,:,j] = b00_pred*float(j) + b10_pred
        else:
            if j==0:
                b1_tmp = b10_pred
                b0_tmp = b00_pred
            else:
                b1_tmp = b1_pred[:,:,j-1]
                b0_tmp = b0_pred[:,:,j-1]
            b0_pred[:,:,j] = b0_tmp + gp_pred(data["bi0"][:,:,j],L_b0,dx2,dx12,data["s_b0"]/1000,data["l_b0"],delta)
            b1_pred[:,:,j] = b1_tmp + b0_tmp
    sigma_pred = gp_pred(np.log(data["sigma"]) - (data["xp"] @ data["beta_s"].T).T,L_s,dx2,dx12,data["s_sigma"],data["l_sigma"],delta)
    sigma_pred  = np.exp((data["xpred"] @ data["beta_s"].T).T + sigma_pred)
    if xi_spatial == 1 and is_gumbel == 0:
        xi_pred = gp_pred(data["xi"],L_xi,dx2,dx12,data["s_xi"],data["l_xi"],delta)
elif nngp == 1:
    b1_pred = np.zeros((nit,npr,no))
    b10_pred = np.zeros((nit,npr))
    b00_pred = np.zeros((nit,npr))
    sigma_pred = np.zeros((nit,npr))
    Ddelta = np.identity(M)*delta
    if not trend_linear:
        b0_pred = np.zeros((nit,npr,no))
        bi0_pred = np.zeros((nit,npr,no))
    b10c = data["b10"] - (data["xp"] @  data["beta_b1"].T).T
    b00c = data["b00"]
    if not trend_linear:
        Q0p = matern_52(dx12,data["s_b0"]/1000,data["l_b0"])
    sigmac = np.log(data["sigma"]) - (data["xp"] @  data["beta_s"].T).T
    Q10p = matern_52(dx12,data["s_b10"],data["l_b10"])
    Q00p = matern_52(dx12,data["s_b00"]/scale_b00,data["l_b00"])
    Qsp = matern_52(dx12,data["s_sigma"],data["l_sigma"])
    if xi_spatial == 1 and is_gumbel == 0:
        xi_pred = np.zeros((nit,npr))
        Qxip = matern_52(dx12,data["s_xi"],data["l_xi"])
    for i in range(npr):
        Q10 = matern_52(dx[:,:,i],data["s_b10"],data["l_b10"]) + Ddelta
        Q00 = matern_52(dx[:,:,i],data["s_b00"]/scale_b00,data["l_b00"]) + Ddelta
        Qs = matern_52(dx[:,:,i],data["s_sigma"],data["l_sigma"]) + Ddelta
        if not trend_linear:
            Q0 = matern_52(dx[:,:,i],data["s_b0"]/1000,data["l_b0"]) + Ddelta
        if xi_spatial == 1 and is_gumbel == 0:
            Qxi = matern_52(dx[:,:,i],data["s_xi"],data["l_xi"]) + Ddelta
        for j in range(nit):
            mi = Q10p[j,:,i] @ solve(Q10[j,:,:],b10c[j,indM[:,i]])
            vi = np.square(data["s_b10"][j]) - Q10p[j,:,i] @ solve(Q10[j,:,:],Q10p[j,:,i])
            b10_pred[j,i] = mi + np.sqrt(vi) * standard_normal()
            
            mi = Q00p[j,:,i] @ solve(Q00[j,:,:],b00c[j,indM[:,i]])
            vi = np.square(data["s_b00"][j]/scale_b00) - Q00p[j,:,i] @ solve(Q00[j,:,:],Q00p[j,:,i])
            b00_pred[j,i] = mi + np.sqrt(vi) * standard_normal()
            
            mi = Qsp[j,:,i] @ solve(Qs[j,:,:],sigmac[j,indM[:,i]])
            vi = np.square(data["s_sigma"][j]) - Qsp[j,:,i] @ solve(Qs[j,:,:],Qsp[j,:,i])
            sigma_pred[j,i] = mi + np.sqrt(vi) * standard_normal()
            
            if not trend_linear:  
                bi0c = data["bi0"][j,:,:]
                mi = Q0p[j,:,i] @ solve(Q0[j,:,:],bi0c[indM[:,i],:])
                vi = np.square(data["s_b0"][j]/1000) - Q0p[j,:,i] @ solve(Q0[j,:,:],Q0p[j,:,i])
                bi0_pred[j,i,:] = mi + np.sqrt(vi) * standard_normal(mi.size)
                
            if xi_spatial == 1 and is_gumbel == 0:
                mi = Qxip[j,:,i] @ solve(Qxi[j,:,:],data["xi"][j,indM[:,i]])
                vi = np.square(data["s_xi"][j]) - Qxip[j,:,i] @ solve(Qxi[j,:,:],Qxip[j,:,i])
                xi_pred[j,i] = mi + np.sqrt(vi) * standard_normal()      
    b10_pred = (data["xpred"] @ data["beta_b1"].T).T + b10_pred
    sigma_pred = np.exp((data["xpred"] @ data["beta_s"].T).T + sigma_pred)
        
    for j in range(no):
        if trend_linear:
            b1_pred[:,:,j] = b00_pred*float(j) + b10_pred
        else:
            if j==0:
                b1_tmp = b10_pred
                b0_tmp = b00_pred
            else:
                b1_tmp = b1_pred[:,:,j-1]
                b0_tmp = b0_pred[:,:,j-1]
            b0_pred[:,:,j] = b0_tmp + bi0_pred[:,:,j]
            b1_pred[:,:,j] = b1_tmp + b0_tmp

if xi_spatial == 0 and is_gumbel == 0:
    xi_star = np.repeat(data["xi"] * data["alpha"],npr,axis=1).flatten('F')[:,None]
    theta = np.power(A_wl,xi_star)
    xi = np.repeat(data["xi"],npr,axis=1).flatten('F')[:,None]
    xi_pred = data["xi"]
elif is_gumbel == 0:
    xi = xi_pred.flatten('F')[:,None]
    xi_star = xi * (np.repeat(data["alpha"],npr,axis=1).flatten('F')[:,None])
    theta = np.power(A_wl,xi_star)

alpha = np.repeat(data["alpha"],npr,axis=1).flatten('F')[:,None]
if is_gumbel:
    sigma_pred_star = alpha * sigma_pred.flatten('F')[:,None]
    sigma_pred_star = np.repeat(sigma_pred_star,no,axis=1)
    mu_pred_star = np.reshape(b1_pred,(nit*npr,no),order='F') + (sigma_pred_star * np.log(A_wl))
    y_pred = gumbel.rvs(mu_pred_star,sigma_pred_star)
else:
    mu_pred_star = np.reshape(b1_pred,(nit*npr,no),order='F') + sigma_pred.flatten('F')[:,None] * (theta - 1)/xi
    sigma_pred_star = alpha * sigma_pred.flatten('F')[:,None] * theta
    y_pred = gev.rvs(-np.repeat(xi_star,no,axis=1),mu_pred_star,sigma_pred_star)
y_pred = np.reshape(y_pred,(nit,npr,no),order='F')

data2 = {}
data2["b1_pred"] = b1_pred
if trend_linear:
    data2["b00_pred"] = b00_pred
    data2["mu_intercept"] = b10_pred
else:
    data2["b0_pred"] = b0_pred
data2["sigma_pred"] = sigma_pred
if not is_gumbel:
    data2["xi_pred"] = xi_pred
data2["y_pred"] = y_pred
data2["y_nan"] = yi
data2["lon_pred"] = data["loni"]
data2["lat_pred"] = data["lati"]

savemat(pathOut + outFile,data2,oned_as='column')
        
