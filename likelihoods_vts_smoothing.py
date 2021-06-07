from scipy.special import factorial
from scipy import stats as ss

import functions
from models import *
############################  Likelihoods  ####################################

def calculate_loglikelihood_UinComov(args_sampler, hyperparams):    
    """ 
    Calculates log-likelihood for an analysis with no redshift evolution of 
        the merger rate (see eq 2 in arXiv:2006.15047).

    Parameters
    ----------
    args_sampler : analysis arguments defined in the analysis file.
    hyperparams : Hyperpameters sampled using the analysis file.

    Returns
    -------
    Log-likelihood
    """
    
    locs_mch, stds_mch = hyperparams['locs_mch'], hyperparams['stds_mch']
    locs_sz, stds_sz = hyperparams['locs_sz'], hyperparams['stds_sz']
    sz_norm = hyperparams['sz_norm']
    #for speed-up change 1-D Gaussians to 3-D Gaussians with no covariance
    var_means, var_covs = functions.get_mchirp_spinz_params(locs_mch, stds_mch, \
                                                                locs_sz, stds_sz)
    min_q, alphas_q = hyperparams['min_q'], hyperparams['alphas_q']
    gwts = hyperparams['gwts']
    rate = hyperparams['rate']
    injections = args_sampler['injections']
    data = args_sampler['data']
    ngauss = args_sampler['ngauss']
    
    log_select, log_poisson, sumvt, vt_err_sq = 0, 0, 0, 0
    for obsrun in injections.keys():
        nobs = len(data[obsrun]['breaks']) - 1
        vt, vtfracerr_orun = get_vt_nosfr(injections[obsrun], var_means, var_covs,\
                                                min_q, alphas_q, gwts, sz_norm)
        log_select += nobs * np.log(vt)
        mu = rate * vt
        log_poisson += nobs * np.log(mu) - mu
            
        sumvt += vt
        vt_err_sq += vt ** 2 * vtfracerr_orun ** 2
    vt_frac_err = np.sqrt(vt_err_sq) / sumvt

    if vt_frac_err > 0.1:
        return -np.inf
    
    logsum_sumprob = 0
    log_sample_size = 0
    for obsrun in data.keys():
        breaks = data[obsrun]['breaks']
        nobs = len(breaks) - 1
        logq =  data[obsrun]['logq']
        parametric_data = data[obsrun]['parametric_data']
        pe_prior_pdf = data[obsrun]['pe_prior_pdf']
        prob_flat = 0
        for ii in range(ngauss):
            logpdfq = powerlaw_logpdf(logq, min_q[ii], 1., alphas_q[ii])
            logprob = core_logpdf(parametric_data, var_means[ii], var_covs[ii], \
                                                                logpdfq, gwts[ii])
            #The Gaussians modeling spins are actually truncated 
            logprob -= np.log(sz_norm[ii]) 
            prob_flat += np.exp(logprob)

        prob_flat /= pe_prior_pdf
        for jj in range(nobs):
            logsum_sumprob += np.log(np.sum(prob_flat[breaks[jj] : breaks[jj + 1]]))
            log_sample_size += np.log((breaks[jj + 1] - breaks[jj]))

    logsum_sumprob -= log_sample_size #normalize per pe sample
    logsum_sumprob -= log_select #correct for selection effect
    logsum_sumprob += log_poisson #extended likelihood

    return logsum_sumprob

############################  VTs  ####################################

def get_vt_nosfr(injections, var_means, var_covs, min_q, alphas_q, gwts, sz_norm):
    """ 
    Calculates the sensitive volume for correction of selection effects and 
        calculation of merger rate (numerator in eq 2 of arXiv:2006.15047).

    Parameters
    ----------
    injections : Dictionary containing injections for all the observation runs
    var_means: Mean values of Gaussians organised on the diagonal
    var_covs: Scale squared of Gaussians organised on a diagonal
    min_q: Refer Table 1. in arXiv:2006.15047
    alphas_q: Refer Table 1. in arXiv:2006.15047
    gwts: Mixing weights for the components
    sz_norm: Normalisation for Gaussians modeling spins
    These are needed as there are no injections outside |sz| > 0.99

    Returns
    -------
    Sensitive volume and it's Poisson error for the given hyper-parameters.
    """
    
    ngauss = len(var_means)
    var_rec = injections['var_rec']
    logq_rec = injections['logq_rec']
    rec_pdf = injections['rec_pdf']
    VT = injections['surveyed_VT']#includes T where triggers were not generated
    ndraw = injections['ndraw']
    
    pout = 0
    for ii in range(ngauss):
        logpdfq = powerlaw_logpdf(logq_rec, min_q[ii], 1., alphas_q[ii])
        logpdf = core_logpdf(var_rec, var_means[ii], var_covs[ii], logpdfq, gwts[ii])
        logpdf -= np.log(sz_norm[ii])
        pout += np.exp(logpdf)
    
    w =  pout / rec_pdf
    sumw = np.sum(w)
    
    return VT * sumw/ndraw, np.sqrt(np.sum(np.square(w))) / sumw

############################  Smoothing  ####################################

def get_mch_AD(ref_mch_samp, ref_mch_prob, locs_mch, stds_mch, gwts):
    """ 
    Calculates the r_eff term defined in eq.5 of arXiv:2006.15047

    Parameters
    ----------
    ref_mch_samp: Bin edges for the reference populations
    ref_mch_prob: Probability contained in each of these bins
    locs_mch: Location of Gaussians modeling the chirp mass
    stds_mch: Scales of Gaussians modeling the chirp mass
    gwts: Mixing weights for the components

    Returns
    -------
    r_eff
    """
    
    nsamp_mch = len(ref_mch_samp)
    ngauss = len(locs_mch)
    prp_mch_prob = 0
    for kk in range(ngauss):
        prp_cdfs = norm.cdf(ref_mch_samp, loc = locs_mch[kk], scale = stds_mch[kk])
        prp_prob = prp_cdfs[1:] - prp_cdfs[:-1]
        prp_mch_prob += prp_prob * gwts[kk]

    w = prp_mch_prob / ref_mch_prob
    w /= np.max(w)
    AD_mch = np.sum(w) / len(w)
    
    return AD_mch

def get_cpn_AD(ref_cpn_samp, ref_cpn_prob, locs_sz, stds_sz, min_q, alphas_q, gwts):
    """ 
    Calculates the r_eff term for chi_pn - Currently not used

    Parameters
    ----------
    ref_cpn_samp: Bin edges for the reference populations
    ref_cpn_prob: Probability contained in each of these bins
    locs_sz: Location of Gaussians modeling the aligned spins
    stds_sz: Scales of Gaussians modeling the aligned spins
    min_q: Refer Table 1. in arXiv:2006.15047
    alphas_q: Refer Table 1. in arXiv:2006.15047
    gwts: Mixing weights for the components

    Returns
    -------
    r_eff
    """
    nsamp = 100000
    ngauss = len(locs_sz)
    nsamp_cpn = len(ref_cpn_samp)
    s1z_samp, s2z_samp, q_samp = [], [], []
    for kk in range(ngauss):
        
        npergauss = int(nsamp  * gwts[kk]) + 1
        s1z_samp = np.append(s1z_samp, ss.norm.rvs(loc = locs_sz[kk], scale = stds_sz[kk], size = npergauss))
        s2z_samp = np.append(s2z_samp, ss.norm.rvs(loc = locs_sz[kk], scale = stds_sz[kk], size = npergauss))
        q_samp = np.append(q_samp, powerlaw_samples(min_q[kk], 1.0, alphas_q[kk], npergauss))
    
    prp_cpn_samp = get_pn1(q_samp, s1z_samp, s2z_samp)
    hist, _ = np.histogram(prp_cpn_samp, bins = ref_cpn_samp, density = True)
    prp_cpn_prob = hist * np.diff(ref_cpn_samp)
        
    AD_cpn = (nsamp_cpn - 1) * np.sum((ref_cpn_prob - prp_cpn_prob) ** 2)
    
    return AD_cpn