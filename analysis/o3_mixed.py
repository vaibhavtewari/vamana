import os
import numpy as np
from copy import deepcopy
from scipy.stats import chi2, multivariate_normal
import models
import proposals, functions

#Analysis specific imports

#Import for function that calculates the log-likelihood(Eq. 2 in in arXiv:2006.15047)
from likelihoods_vts_smoothing import calculate_loglikelihood_sfr_onepluszm1m2 as loglikelihood
#Import function that organises data
import analysis_data
#Import for function that calculates posterior predictive p(\theta|\Lambda)
from post_process import post_process as postprocess

#Power law to make location proposals in chirp mass(please refer to fig.1 in arXiv:2006.15047)
min_mass, max_mass = 5, 75.
alpha = 2.0
ngauss = 10 #Number of components in the mixture
dcdf = 0.015 #Equation 7 in arXiv:2006.15047

sz = np.linspace(-0.8, 0.8, 161)
dsz = sz[1] - sz[0]
p_sz = np.zeros_like(sz)
idx = np.where(np.abs(sz) > 0.2)
p_sz[idx] = 1 / np.abs(sz[idx])
idx = np.where(0.2 >= np.abs(sz))
p_sz[idx] = np.max(p_sz) * np.ones_like(sz[idx])
c_sz = np.cumsum(p_sz)
c_sz /= c_sz[-1]

k = np.linspace(-5.0, 5.0, 401)
dk = k[1] - k[0]
p_k = np.zeros_like(k)
idx = np.where(np.abs(k) > 2.0)
p_k[idx] = 1 / np.abs(k[idx])
idx = np.where(2.0 >= np.abs(k))
p_k[idx] = np.max(p_k) * np.ones_like(k[idx])
c_k = np.cumsum(p_k)
c_k /= c_k[-1]

#Following function is called from script.py
#Define: range of priors, the axis to calculate posterior predictive
# Number of interation, posterior collection starting at which iteration
# Collect nstride'th posterior
def define_args(pe_orig, inj_orig):
    
    pe, injections = deepcopy(pe_orig), deepcopy(inj_orig)
    args_sampler = {}
    args_sampler['ngauss'] = ngauss
    data, injections = analysis_data.get_data_for_fitting(pe, injections)
    args_sampler['data'] = data
    args_sampler['injections'] = injections
    args_sampler['nobs'] = sum(len(pe[obsrun].keys()) for obsrun in pe.keys())
    
    args_sampler['max_dcdf'] = dcdf
    args_sampler['rate_range'] = [1., 100.]
    
    min_nu, max_nu, nsamp = int(2 / dcdf ** 2 / 4), int(4 * 2 / dcdf ** 2), 10000
    dof_prior = functions.get_dof(min_nu, max_nu, nsamp)
    args_sampler['dof_prior'] = dof_prior
    
    args_sampler['min_mass'] = min_mass
    args_sampler['max_mass'] = max_mass
    args_sampler['alpha'] = alpha

    args_sampler['corr_m1m2_range'] = [-0.75, 0.75]
    args_sampler['d_corr_m1m2'] = 0.01
    
    args_sampler['kappa_range'] = np.array([-4, 4])
    args_sampler['d_kappa'] = 0.05
    
    args_ppd = {}
    args_ppd['mass_ax'] = np.linspace(1.0, 101.0, 1001)
    args_ppd['q_ax'] = np.linspace(0.1, 1., 50)
    args_ppd['sz_ax'] = np.linspace(-1, 1, 100)
    args_ppd['mch_ax'] = np.linspace(1.0, 76.0, 751)
    
    args_ppd['niter'] = 20000
    args_ppd['nstart'] = 10000
    args_ppd['nstride'] = 200
    args_ppd['output'] = 'temp'
    if not os.path.isdir(args_ppd['output']):
        os.makedirs(args_ppd['output'])
    
    return args_sampler, args_ppd

# Following function is called from function_gauss in functions.py
# It initialises the hyper-parameters
def initialise_hyperparams(args_sampler):
    
    ngauss = args_sampler['ngauss']
    locs_m1 = np.exp(np.linspace(np.log(min_mass + 1), np.log(max_mass), ngauss))
    locs_m2 = np.maximum(min_mass, locs_m1 * np.random.uniform(0.7, 1.0, ngauss))
    hyperparams = {}
    stds_m1, stds_m2 = [], []
    for ii in range(ngauss):
        std_m1 = 0.13 * locs_m1[ii]
        stds_m1.append(std_m1)
        std_m2 = 0.13 * locs_m2[ii]
        stds_m2.append(std_m2)
    stds_m1 = np.array(stds_m1)
    stds_m2 = np.array(stds_m2)
    
    hyperparams['locs_m1'] = locs_m1
    hyperparams['stds_m1'] = stds_m1
    hyperparams['locs_m2'] = locs_m2
    hyperparams['stds_m2'] = stds_m2
    hyperparams['corr_m1m2'] = np.zeros(ngauss)
    
    rate_range = args_sampler['rate_range']
    l, h = rate_range[0], rate_range[1]
    hyperparams['rate'] = np.random.uniform(l, h)
    hyperparams['gwts'] = 1/locs_m1
    hyperparams['gwts'] /= np.sum(hyperparams['gwts'])
    
    hyperparams['locs_sz'] = np.zeros(ngauss)
    hyperparams['stds_sz'] = 0.2 * np.ones(ngauss)

    k = 2.0
    hyperparams['kappa_single'] = k
    hyperparams['kappa'] = np.random.uniform(k - 1, k + 1, ngauss)
    
    return hyperparams

# Following function is called from function_gauss in functions.py
# It is the place to define the functionals to model the population
# The functionals are defined in proposals.py
# It proposes the new hyper-parameters.
def get_hyperparams_proposal(args_sampler, hyperparams):
    
    prp_hyperparams = {}
    prp_hyperparams['gwts'], mhr_gwts = proposals.get_mixing_fractions(hyperparams['gwts'], \
                                                                    args_sampler['dof_prior'], ngauss)
    
    prp_hyperparams['locs_m1'], prp_hyperparams['locs_m2'], mhr_locs_m1m2 = proposals.get_proposal_mass_locs( \
                                             hyperparams['locs_m1'], hyperparams['locs_m2'], \
                                             args_sampler['max_dcdf'], args_sampler['min_mass'], \
                                             args_sampler['max_mass'], args_sampler['alpha'])
    prp_hyperparams['stds_m1'], mhr_stds_m1 = proposals.get_proposal_mass_stds( \
                                             hyperparams['locs_m1'], prp_hyperparams['locs_m1'], \
                                             hyperparams['stds_m1'], args_sampler['dof_prior'], 0.04, 0.16, 1.0)
    prp_hyperparams['stds_m2'], mhr_stds_m2 = proposals.get_proposal_mass_stds( \
                                             hyperparams['locs_m2'], prp_hyperparams['locs_m2'], \
                                             hyperparams['stds_m2'], args_sampler['dof_prior'], 0.04, 0.16, 1.0)
    mhr_m1m2 = mhr_locs_m1m2 * mhr_stds_m1 * mhr_stds_m2
    
    d_corr = args_sampler['d_corr_m1m2']
    prp_hyperparams['corr_m1m2'] = proposals.get_proposal_uniform(hyperparams['corr_m1m2'], \
                                              args_sampler['corr_m1m2_range'], d_corr)
    
    prp_hyperparams['norm_m1m2'] = []
    for ii in range(ngauss):
        mean = [prp_hyperparams['locs_m1'][ii], prp_hyperparams['locs_m2'][ii]]
        c = prp_hyperparams['stds_m1'][ii] * prp_hyperparams['stds_m2'][ii] * prp_hyperparams['corr_m1m2'][ii]
        cov = [[prp_hyperparams['stds_m1'][ii] ** 2, c],[c, prp_hyperparams['stds_m2'][ii] ** 2]]

        x = multivariate_normal.rvs(mean=mean, cov=cov, size=100000)
        m1, m2 = x.T
        prp_hyperparams['norm_m1m2'] = np.append(prp_hyperparams['norm_m1m2'], len(m2[m2/m1 < 1]) / len(m2))
    
    cdf_sz = np.interp(hyperparams['locs_sz'], sz, c_sz)
    prp_cdf = proposals.get_proposal_uniform(cdf_sz, np.array([0., 1.0]), args_sampler['max_dcdf'])
    prp_hyperparams['locs_sz'] = np.interp(prp_cdf, c_sz, sz)
    
    prp_hyperparams['stds_sz'], prp_hyperparams['norm_sz'], mhr_sz = \
                                        proposals.get_proposal_spinz(prp_hyperparams['locs_sz'], \
                                        hyperparams['stds_sz'], args_sampler['dof_prior'], \
                                        args_sampler['max_dcdf'])
    
    prp_hyperparams['rate'] = proposals.get_proposal_uniforminlog(hyperparams['rate'], \
                                                    args_sampler['rate_range'], args_sampler['max_dcdf'])
    
    cdf_k = np.interp(hyperparams['kappa_single'], k, c_k)
    prp_cdf = proposals.get_proposal_uniform(cdf_k, np.array([0., 1.0]), args_sampler['max_dcdf'])
    prp_hyperparams['kappa_single'] = np.interp(prp_cdf, c_k, k)
    
    klo = hyperparams['kappa_single'] - 1
    khi = hyperparams['kappa_single'] + 1
    f = (hyperparams['kappa'] - klo) / (khi - klo)
    new_klo = prp_hyperparams['kappa_single'] - 1
    new_khi = prp_hyperparams['kappa_single'] + 1
    scaled_k = new_klo + f * (new_khi - new_klo)
    d_k = 2 * args_sampler['max_dcdf']
    prp_hyperparams['kappa'] = proposals.get_proposal_uniform(scaled_k, [new_klo, new_khi], d_k)
    
    mhr = mhr_m1m2 * mhr_sz * mhr_gwts
    
    return prp_hyperparams, mhr