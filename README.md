# BAYEX: Spatiotemporal Bayesian hierarchical modeling of extremes with max-stable processes

BAYEX offers spatiotemporal Bayesian hierarchical modeling of extremes using max-stable and latent processes. As key features, BAYEX makes estimates of both the GEV parameters (location, scale and shape) and the annual maxima at any arbitrary location, either gauged or ungauged, while providing realistic uncertainty estimates. Inference in BAYEX is performed using Hamiltonian Monte Carlo as implemented by the Stan probabilistic programming language. 

## Citation
This version 2.1 (June 2021) of the code has the same features as version 2.0, as described below, but fixes a bug (see below). **The bug does not affect version 1.0 of the code**

Please cite the following paper when using this code:
Calafat, F. M., and M. Marcos (2020). Probabilistic reanalysis of storm surge extremes in Europe, Proc. Natl. Acad. Sci. U. S. A., 117 (4) 1877-1883. 

**bug fix**:
- In version 2.0 covariates were centered prior to fitting the model but they were not reverted back properly for prediction. This led to slightly biased estimates of the time-mean GEV
location and sigma parameters at prediction sites. It does not affect estimates at data sites. I thank D.J. Rasmussen (Princeton University) and Bob Kopp (Rutgers University) for
spotting this.

## New features (as in v2.0)
- Users can now choose to estimate the GEV shape parameter from the data or to set its value to zero effectively assuming a Gumbel distribution

- Allow the GEV shape parameter to vary in space.

- Users can now model temporal variability in the GEV location parameter using either an integrated random walk or a linear trend.

- Add Nearest Neighbor Gaussian Process (NNGP) for the GEV parameters. Users can now choose to fit the model using either full Gaussian processes (GPs) or NNGPs. Full GPs require ~O(n^3) flops and ~O(n^2) storage (n: number of data sites), and so they can rapidly become computationally infeasible. The total flop count per interation in NNGPs is linear in the number of data sites and NNGPs do not require storing or inverting large matrices. This means that the users are now able to fit the model to larger datasets. The implementation of the NNGP is based on the formulation given by Datta et al. (2016) (Datta, A., S. Banerjee, A. O. Finley, and A. E. Gelfand 2016. Hierarchical Nearest-Neighbor Gaussian Process Models for Large Geostatistical Datasets. J. Am. Stat. Assoc. 111, 800–812).

- Add Nearest Neighbor approximation for the residual spatial process. Users can now also choose to build the residual spatial process at each site using either all spatial knots or a smaller subset of knots. The latter further reduces the computational complexity of the model.

- Performance optimizations.

**Note**: when using NNGPs, the predictions for the GEV parameters made by **max_stable_prediction.py** are samples from the marginal posterior distribution (i.e., the GEV parameters at the i-th unobserved location come from **p(x_i | Y)** NOT from the joint posterior **p(x_1, x_2, ..., x_n | Y)**). This is only an issue if one wants to compute spatial averages because in that case the standard deviation of the (spatially averaged) posterior draws will tend to underestimate the true uncertainty of the averaged quantity.

## Deprecations
- Remove the option of making predictions at unobserved location/times within Stan. In this version, predicions can only be made outside Stan. In the previous version, a Matlab script was provided to make predictions offline. This Matlab script is now replaced with a Python script, so that the code is no longer dependent on any commercially licensed software. 
 
## Prerequisites
This code requires:

- PyStan (https://pystan.readthedocs.io/en/latest/)
- Python and various dependencies (Numpy and Scipy)

## How to use BAYEX
The file **maxStable.py** contains the Stan code for the Bayesian Hierarchical model. You might need to edit these files if, for example, you want to try different priors (in the model{} block).

The file **master_maxStable.py** shows how to load and prepare the input data, compile the Stan code, fit the hierarchical model, save the output to a file, and make predictions. Take this file as an example of how to use the code. 

Predictions of the GEV parameters and the annual maxima can be done using the provided Python script (**max_stable_prediction.py**), once the solutions from the Stan model are available.

The key output variables are (the first dimension in all arrays represent MCMC samples from the posterior distribution):
 'xi': GEV shape parameter
 'sigma': GEV scale parameter
 'b1': GEV location parameter
 'b0': instantaneous trend of the GEV location parameter (when using an integrated random walk)
 'b00': trend of the GEV location parameter (when using a linear trend)
 'y_nan': predictions of the annual maxima at missed times at gauged locations.

 variables suffixed with the word 'pred' contain the predictions at ungauged locations:
 'sigma_pred': predictions of the GEV scale parameter at ungauged locations
 'b1_pred': predictions of the GEV location parameter at ungauged locations
 'b0_pred': predictions of the instantaneous trend of the GEV location parameter at ungauged locations (when using an integrated random walk)
 'b00_pred': predictions of the trend of the GEV location parameter at ungauged locations (when using a linear trend)
 'y_pred': predictions of the annual maxima at ungauged locations
 
 **Note**: You are welcome to contact me to report bugs or make suggestions regarding new features

## License

Copyright (C) 2020 Francisco Mir Calafat.

Distributed under GPLv3 as described in LICENSE.txt.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.


