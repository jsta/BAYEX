import numpy as np
from scipy.io import savemat
    
def save_estimates(fit,modelData,pathOut,fileName):
    
    pars = ['xi','sigma','alpha','tau','beta_b1','beta_s','s_b10','s_b0','s_b00','s_sigma','l_b10','l_b0','l_b00','l_sigma',\
            'b1','b10','b0','b00','A','bi0','mu_star','sigma_star','xi_star','s_xi','l_xi']
    data = {}
    data["lon"] = modelData["lon_data"]
    data["lat"] = modelData["lat_data"]
    data["xp"] = modelData["xp"]
    data["lonk"] =  modelData["lon_knot"]
    data["latk"] =  modelData["lat_knot"]
    data["loni"] = modelData["lon_pred"]
    data["lati"] = modelData["lat_pred"]
    data["xpred"] = modelData["xpred"]
    data["y"] = modelData["y"]
    data["nngp"] = modelData["nngp"]
    data["nnknot"] = modelData["nnknot"]
    data["xi_spatial"] = modelData["xi_spatial"]
    data["M"] = modelData["Mgp"]
    data["Mk"] = modelData["Mknot"]
    data["is_gumbel"] = modelData["is_gumbel"]
    data["trend_linear"] = modelData["trend_linear"]
    
    pars_all = fit.sim["pars_oi"] # all parameters
    indPar = [x for x,y in enumerate(pars_all) if str(y) in pars]
    dims_all = fit.sim["dims_oi"]
    dims = [dims_all[i] for i in indPar]
    pars_in = [pars_all[i] for i in indPar]
    nit = fit.sim["iter"] - fit.sim["warmup"]
    nchains = fit.sim["chains"]
    fn = fit.sim["fnames_oi"]
    
    noi,nxi = data["y"].shape
    
    for i,par in enumerate(pars_in):
        dimi = dims[i]
        nx = 1
        for i in range(len(dimi)):
            nx = nx*dimi[i]
        qsize = [nit*nchains,nx]
        qsizei = [qsize[0]]
        qsizei.extend(dimi)
        q = np.zeros(qsize)
        fni = [x for x in fn if par == x.split("[")[0]]
        for j,fnii in enumerate(fni):
            for k in range(nchains):
                f = fit.sim['samples'][k].chains
                i0 = k*nit
                q[i0:(i0 + nit),j] = f[fnii][-nit:]
        q = np.reshape(q,qsizei,order='F')
        if par == "b1" or par == "mu_star" or par == "sigma_star":
            q = np.reshape(q,(q.shape[0],nxi,noi),order='F')
        data[par] = q
    savemat(pathOut + fileName,data,oned_as='column')


