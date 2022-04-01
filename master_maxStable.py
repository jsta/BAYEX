import numpy as np
import maxStable
from prepare_data import prepare_data
from saveEstimates import save_estimates

path2data = '/path/to/data/' # insert path to data
pathOut = '/path/to/output/' # path where the output of the Stan model will be saved
outFile = 'max_stable_estimates.mat' # name of the file where the output from the Stan model will be saved (mat format)


y = np.loadtxt(path2data + 'surges.txt') # T x N matrix containing the annual maxima (T: temporal samples; N: locations)
lon = np.loadtxt(path2data + 'lon.txt') # N x 1 vector containing the longitudes of the data sites
lat = np.loadtxt(path2data + 'lat.txt') # N x 1 vector containing the latitudes of the data sites
x = np.loadtxt(path2data + 'covariates.txt') # N x nc matrix containing the covariates at data sites (nc: number of covariates).
lon_knot = np.loadtxt(path2data + 'lon_knot.txt') # M x 1 vector containing the longitudes of the spatial knots
lat_knot = np.loadtxt(path2data + 'lat_knot.txt') # M x 1 vector containing the latitudes of the spatial knots
lon_pred = np.loadtxt(path2data + 'lon_pred.txt') # P x 1 vector containing the longitudes of the prediction sites
lat_pred = np.loadtxt(path2data + 'lat_pred.txt') # P x 1 vector containing the latitudes of the prediction sites
xpred = np.loadtxt(path2data + 'covariates_pred.txt') # P x nc matrix containing the covariates at prediction sites

# the following function is used to prepare the data and set the options for the Stan model. Options are:
#    'is_gumbel' specifies whether to treat the GEV shape parameter as unknown (0) or set its value to zero (1) (Gumbel distribution) 
#    'xi_spatial' specifies whether to treat the GEV shape parameter as constant (0) or spatially variable (1)
#    'trend_linear' specifies whether to model the GEV location parameter as an integrated random walk (0) or as a linear trend
#     plus and intercept (1) (in the latter case both the trend coefficient and the intercept are modeled as spatial processes)
#    'nngp' specifies whether to use full GPs (0) or NNGPs (1) for the GEV parameters.
#    'nnknot' specifies whether to use all spatial knots (0) or only nearest neighbors (1) when constructing the residual process
#    'Mgp' is the number of nearest neighbors for the NNGPs. Testing indicates that estimates from NNGP models with as few as 5 or 10
#     neighbors are almost identical to estimates from full GP models.
#    'Mknot' is the number of nearest neighbors for the residual process.
modelData = prepare_data(y,lon,lat,lon_knot,lat_knot,x,is_gumbel=0,xi_spatial=0,trend_linear=0,nngp=0,nnknot=0,Mgp=10,Mknot=10,\
            lon_pred=lon_pred,lat_pred=lat_pred,xpred=xpred)
            
sm = maxStable.max_stable_model() # compile the model

control = {'adapt_delta':0.8,'max_treedepth':10} # MCMC sampler parameters
fit = sm.sampling(data=modelData,iter=1500,warmup=1000,chains=4,init_r=0.5,control=control) # fit the model

# The variable 'fit' is a stanfit object that contains the output derived from fitting the max-stable model to the data
# For convenience, a Python function is provided that extracts the data in 'fit' and saves them to a file:
save_estimates(fit,modelData,pathOut,outFile)

# Once the solutions from the Stan model are available, predictions at unobserved locations/times can be made by running the Python script: max_stable_prediction.py



