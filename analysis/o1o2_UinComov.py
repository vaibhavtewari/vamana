import numpy as np
import models
import proposals, functions, analysis_data
from copy import deepcopy

#Analysis specific imports

#Import for function that calculates r_eff
from likelihoods_vts_smoothing import get_mch_AD as get_mch_AD
#Import for function that calculates the log-likelihood(Eq. 2 in in arXiv:2006.15047)
from likelihoods_vts_smoothing import calculate_loglikelihood_UinComov as loglikelihood
#Import for function that calculates the posterior-predictive
from post_process import post_process_UinComov as postprocess

#Best fit parametric model
#The ends/slope of the powerlaw for the reference chirp mass distribution
min_mch_ref, max_mch_ref = 7.709, 37.847
alpha_mch_ref = 2.284
#The ends/slope of the powerlaw that makes proposals for the locations of Gaussians
#modeling the chirpmass - this is defined same as the referencd distribution
#but can be anything similar (see equation 7.)
min_mch, max_mch = 7.0, 40.
alpha_mch = 1.8
#Number of components used by the analysis
ngauss = 5
#Delta F in equation 7.
dcdf = 0.08

def define_args(pe_orig, inj_orig):
    #Define hyperparameter ranges
    #Define arguments for posterior predictive
    #Define r_eff(AD_thr_mch) value - set to zero to disable broadband smoothing
    pe, injections = deepcopy(pe_orig), deepcopy(inj_orig)
    
    args_sampler = {}
    args_sampler['ngauss'] = ngauss
    pe, injection = analysis_data.reweight_data_to_UinComov(pe, injections)
    data, injections = analysis_data.get_data_for_fitting(pe, injections)
    
    args_sampler['data'] = data
    args_sampler['injections'] = injections
    
    args_sampler['max_dcdf_mch'] = dcdf
    args_sampler['max_dcdf_sz'] = dcdf
    args_sampler['qmin_prior'] = [0., 0.95]
    args_sampler['d_qmin'] = dcdf * np.diff(args_sampler['qmin_prior'])
    args_sampler['alphaq_prior'] = [-7., 1.]
    args_sampler['d_alphaq'] = dcdf * np.diff(args_sampler['alphaq_prior'])
    args_sampler['sz_prp_loc_scale'] = [0.0, 0.3]
    
    args_sampler['max_spin'] = 0.99
    args_sampler['AD_thr_mch'] = 0.1
    
    args_sampler['rate_range'] = [1., 100.]
    args_sampler['d_rate'] = dcdf * np.mean(args_sampler['rate_range'])
    
    min_nu, max_nu, nsamp = int(2 / dcdf ** 2 / 4), int(4 * 2 / dcdf ** 2), 10000
    dof_prior = functions.get_dof(min_nu, max_nu, nsamp)
    #Not really a prior but functions can randomly choose dof from this set to make proposals
    args_sampler['dof_prior'] = dof_prior
    
    args_sampler['alpha'] = alpha_mch
    args_sampler['min_mch'] = min_mch
    args_sampler['max_mch'] = max_mch
    
    #Bin the density for the regerence distribution - eq.5 in arXiv:2006.15047
    cdfs_mch = np.linspace(0., 1.0, 1001)
    ref_mch_samp = np.array([models.powerlaw_invcdf(c, min_mch_ref, max_mch_ref, alpha_mch_ref) for c in cdfs_mch])
    ref_mch_prob = cdfs_mch[1:] - cdfs_mch[:-1]
    args_sampler['ref_mch_samp'] = ref_mch_samp
    args_sampler['ref_mch_prob'] = ref_mch_prob
    
    # Various axes for posterior predictive
    args_ppd = {}
    min_comp, max_comp = 1., 100.
    args_ppd['mass_ax'] = np.linspace(min_comp, max_comp, 750)
    args_ppd['q_ax'] = np.linspace(0.1, 1., 50)
    args_ppd['sz_ax'] = np.linspace(-1, 1, 100)
    args_ppd['mch_ax'] = np.linspace(0, 80.,800)
    
    #Iteration at first sample collection and stride between selected samples
    #These are very large to remove any autocorrelation
    #However the stride can be reduced according to the ACF
    args_ppd['niter'] = 110000
    args_ppd['nstart'] = 5000
    args_ppd['nstride'] = 500
    args_ppd['nppd_per_posterior'] = 500
    #Folder to save the posterior 
    args_ppd['output'] = 'temp/'
    if args_ppd['output'][-1] != '/':
        args_ppd['output'] += '/'
        
    return args_sampler, args_ppd

def initialise_hyperparams(args_sampler):
    #Initialise the hyperparameters
    
    ngauss = args_sampler['ngauss']
    nsamp = len(args_sampler['ref_mch_samp'])
    idx = np.linspace(0.1 * nsamp, 0.9 * nsamp, ngauss).astype(int)
    locs_mch = args_sampler['ref_mch_samp'][idx]
    hyperparams, stds_mch = {}, []
    for ii in range(ngauss):
        std_mch = 0.5 * np.sqrt(15/ngauss) * 0.1 * locs_mch[ii]
        stds_mch.append(std_mch)
    stds_mch = np.array(stds_mch)
    
    hyperparams['locs_mch'] = locs_mch
    hyperparams['stds_mch'] = stds_mch
    
    qmin_prior = args_sampler['qmin_prior']
    alphaq_prior = args_sampler['alphaq_prior']
    hyperparams['min_q'] = np.array([qmin_prior[0]] * ngauss)
    hyperparams['alphas_q'] = np.random.uniform(alphaq_prior[0], alphaq_prior[1], ngauss)
    rate_range = args_sampler['rate_range']
    hyperparams['rate'] = np.random.uniform(rate_range[0], rate_range[1])
    hyperparams['gwts'] = np.array(ngauss * [1./ngauss])
    
    hyperparams['locs_sz'] = np.array(ngauss * [0.0])
    hyperparams['stds_sz'] = np.array(ngauss * [0.2])
    
    return hyperparams

def get_hyperparams_proposal(args_sampler, hyperparams):
    #Function that takes in current hyperparameters 
    #and makes a proposal 
    
    prp_hyperparams = {}
    prp_hyperparams['locs_mch'], prp_hyperparams['stds_mch'], mhr_mch = proposals.get_proposal_mchirp( \
                                        hyperparams['locs_mch'], hyperparams['stds_mch'], \
                                        args_sampler['dof_prior'], args_sampler['max_dcdf_mch'], \
                                        args_sampler['min_mch'], args_sampler['max_mch'], \
                                        args_sampler['alpha'])
    prp_hyperparams['locs_sz'], prp_hyperparams['stds_sz'], prp_hyperparams['sz_norm'], mhr_sz = \
                                        proposals.get_proposal_spinz(hyperparams['locs_sz'], \
                                        hyperparams['stds_sz'], args_sampler['dof_prior'], \
                                        args_sampler['sz_prp_loc_scale'], args_sampler['max_dcdf_sz'])
    prp_hyperparams['min_q'], prp_hyperparams['alphas_q'] = proposals.get_proposal_q_pl( \
                                        hyperparams['min_q'], hyperparams['alphas_q'], \
                                        args_sampler['qmin_prior'], args_sampler['alphaq_prior'], \
                                        args_sampler['d_qmin'], args_sampler['d_alphaq'])
    
    prp_hyperparams['gwts'], mhr_gwts = proposals.get_mixing_fractions(hyperparams['gwts'], \
                                                                    args_sampler['dof_prior'], ngauss)
    
    prp_hyperparams['rate'], mhr_rate = proposals.get_proposal_uniforminlog(hyperparams['rate'], \
                                                    args_sampler['rate_range'], args_sampler['d_rate'])
            
    AD_mch = get_mch_AD(args_sampler['ref_mch_samp'], args_sampler['ref_mch_prob'], \
                        prp_hyperparams['locs_mch'], prp_hyperparams['stds_mch'], \
                        prp_hyperparams['gwts'])
    
    mhr = mhr_mch * mhr_sz * mhr_gwts * mhr_rate
    
    return prp_hyperparams, mhr, AD_mch