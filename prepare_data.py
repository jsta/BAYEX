import numpy as np

def prepare_data(y,lon_data,lat_data,lon_knot,lat_knot,xp,is_gumbel,xi_spatial,trend_linear,nngp,nnknot,Mgp=10,Mknot=10,\
                 lon_pred=np.zeros((0,)),lat_pred=np.zeros((0,)),xpred=np.zeros((0,0)),grainsize=1):
    
    nc = xp.shape[1]
    no,nx = y.shape
    
    y_flat = y.flatten('C') # row-major order (matrices in Stan store data in column-major order)
    kok = np.nonzero(y_flat == y_flat)[0]
    knan = np.nonzero(y_flat != y_flat)[0]
    y_ok = y_flat[kok]
    y_nan = y_flat[knan]
    Nok = y_ok.shape[0]
    Nnan = y_nan.shape[0]
    kok = kok + 1 # Python uses zero-based indexing while in Stan indices are one-based
    knan = knan + 1
    
    if nnknot == 1:
        for i in range(nx):
            dxi = np.sqrt(np.power(lon_knot - lon_data[i],2) + np.power(lat_knot - lat_data[i],2))/10.
            if i == 0:
                indk = np.argsort(dxi)[0:Mknot]
            else:
                indki = np.argsort(dxi)[0:Mknot]
                indk = np.concatenate((indk,indki))
        indk = np.unique(indk)
        lon_knot = lon_knot[indk]
        lat_knot = lat_knot[indk]
    nl = lon_knot.shape[0]
    if nnknot == 0:
        Mknot = 0
    if nngp == 0:
        Mgp = 0
    
    modelData = {'no':no,'nx':nx,'Nok':Nok,'kok':kok,'y_ok':y_ok,'nl':nl,'nc':nc,'nngp':nngp,'lon_pred':lon_pred,'lat_pred':lat_pred,\
                 'xpred':xpred,'xp':xp,'lon_data':lon_data,'lat_data':lat_data,'lon_knot':lon_knot,'lat_knot':lat_knot,'Mgp':Mgp,\
                 'grainsize':grainsize,'xi_spatial':xi_spatial,'nnknot':nnknot,'Mknot':Mknot,'y':y,'is_gumbel':is_gumbel,\
                 'trend_linear':trend_linear}
                
    return modelData


