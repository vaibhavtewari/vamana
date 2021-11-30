import os
import numpy as np
from copy import deepcopy
from scipy.stats import chi2
import models
import proposals, functions

#Analysis specific imports
from likelihoods_vts_smoothing import calculate_loglikelihood_sfr_onepluszmch as loglikelihood
import analysis_data
from post_process import post_process as postprocess

#Best fit parametric model
min_mch, max_mch = 5.2, 67.
alpha_mch = 2.5
ngauss = 11
dcdf = 0.02

def define_args(pe_orig, inj_orig):
    
    pe, injections = deepcopy(pe_orig), deepcopy(inj_orig)
    args_sampler = {}
    args_sampler['ngauss'] = ngauss
    data, injections = analysis_data.get_data_for_fitting(pe, injections)
    args_sampler['data'] = data
    args_sampler['injections'] = injections
    args_sampler['nobs'] = sum(len(pe[obsrun].keys()) for obsrun in pe.keys())
    
    args_sampler['max_dcdf'] = dcdf
    args_sampler['qmin_prior'] = [0.1, 0.95]
    args_sampler['d_qmin'] = dcdf * np.diff(args_sampler['qmin_prior'])
    args_sampler['alphaq_prior'] = [-7., 2.]
    args_sampler['d_alphaq'] = dcdf * np.diff(args_sampler['alphaq_prior'])
    args_sampler['sz_prp_loc_scale'] = [0.0, 0.5]
    
    args_sampler['rate_range'] = [1., 100.]
    args_sampler['d_rate'] = dcdf * np.diff(args_sampler['rate_range'])[0]
    
    min_nu, max_nu, nsamp = int(2 / dcdf ** 2 / 4), int(4 * 2 / dcdf ** 2), 10000
    dof_prior = functions.get_dof(min_nu, max_nu, nsamp)
    args_sampler['dof_prior'] = dof_prior
    
    args_sampler['alpha'] = alpha_mch
    args_sampler['min_mch'] = min_mch
    args_sampler['max_mch'] = max_mch
    
    args_sampler['kappa_range'] = np.array([-1, 4])
    args_sampler['d_kappa'] = dcdf * np.diff(args_sampler['kappa_range'])[0]
    
    args_ppd = {}
    min_comp, max_comp = 1., 100.
    args_ppd['mass_ax'] = np.linspace(min_comp, max_comp, 750)
    args_ppd['q_ax'] = np.linspace(0.1, 1., 50)
    args_ppd['sz_ax'] = np.linspace(-1, 1, 100)
    args_ppd['mch_ax'] = np.linspace(0, 80.,800)
    
    args_ppd['niter'] = 50000
    args_ppd['nstart'] = 10000
    args_ppd['nstride'] = 500
    args_ppd['nppd_per_posterior'] = 500
    args_ppd['output'] = 'temp'
    if not os.path.isdir(args_ppd['output']):
        os.makedirs(args_ppd['output'])
    
    return args_sampler, args_ppd

def initialise_hyperparams(args_sampler):
    
    ngauss = args_sampler['ngauss']
    locs_mch = np.exp(np.linspace(np.log(min_mch) + 0.1, np.log(max_mch) - 0.1, ngauss))
    hyperparams, stds_mch = {}, []
    for ii in range(ngauss):
        std_mch = 0.1 * np.sqrt(15/ngauss) * locs_mch[ii]
        stds_mch.append(std_mch)
    stds_mch = np.array(stds_mch)
    
    hyperparams['locs_mch'] = locs_mch
    hyperparams['stds_mch'] = stds_mch
    
    qmin_prior = args_sampler['qmin_prior']
    hyperparams['min_q'] = np.array([qmin_prior[0]] * ngauss)
    hyperparams['alphas_q'] = -2 * np.ones(ngauss)
    rate_range = args_sampler['rate_range']
    l, h = rate_range[0], rate_range[1]
    hyperparams['rate'] = np.random.uniform(l, h)
    hyperparams['gwts'] = 1/locs_mch ** alpha_mch
    hyperparams['gwts'] /= np.sum(hyperparams['gwts'])
    
    hyperparams['locs_sz'] = np.zeros(ngauss)
    hyperparams['stds_sz'] = 0.2 * np.ones(ngauss)
    
    hyperparams['kappa'] = np.ones(ngauss)
    
    return hyperparams
    
def get_hyperparams_proposal(args_sampler, hyperparams):
    
    prp_hyperparams = {}
    prp_hyperparams['locs_mch'], prp_hyperparams['stds_mch'], mhr_mch = proposals.get_proposal_mchirp( \
                                        hyperparams['locs_mch'], hyperparams['stds_mch'], \
                                        args_sampler['dof_prior'], args_sampler['max_dcdf'], \
                                        args_sampler['min_mch'], args_sampler['max_mch'], \
                                        args_sampler['alpha'])
    
    prp_hyperparams['locs_sz'], prp_hyperparams['stds_sz'], prp_hyperparams['sz_norm'], mhr_sz = \
                                        proposals.get_proposal_spinz(hyperparams['locs_sz'], \
                                        hyperparams['stds_sz'], args_sampler['dof_prior'], \
                                        args_sampler['sz_prp_loc_scale'], args_sampler['max_dcdf'])
    
    prp_hyperparams['min_q'] = proposals.get_proposal_uniform(hyperparams['min_q'], \
                                                    args_sampler['qmin_prior'], args_sampler['d_qmin'])
    prp_hyperparams['alphas_q'] = proposals.get_proposal_uniform(hyperparams['alphas_q'], \
                                                args_sampler['alphaq_prior'], args_sampler['d_alphaq'])
    
    prp_hyperparams['gwts'], mhr_gwts = proposals.get_mixing_fractions(hyperparams['gwts'], \
                                                                    args_sampler['dof_prior'], ngauss)
    
    prp_hyperparams['rate'] = proposals.get_proposal_uniforminlog(hyperparams['rate'], \
                                                    args_sampler['rate_range'], args_sampler['max_dcdf'])
    
    prp_hyperparams['kappa'] = proposals.get_proposal_uniform(hyperparams['kappa'], \
                                              args_sampler['kappa_range'], args_sampler['d_kappa'])
    
    mhr = mhr_mch * mhr_sz * mhr_gwts
    
    return prp_hyperparams, mhr