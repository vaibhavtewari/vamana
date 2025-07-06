from scipy.special import factorial
from scipy import stats as ss

import functions
from models import *
############################  Likelihoods  ####################################

def calculate_loglikelihood_sfr_onepluszm1m2(args_sampler, hyperparams):
    """ 
    Calculates log-likelihood for an analysis that has power-law 
    redshift evolution for all or individual components. For details see
    https://arxiv.org/abs/2012.08839

    Parameters
    ----------
    args_sampler : analysis arguments defined in the analysis file.
    hyperparams : Hyperpameters sampled using the analysis file.

    Returns
    -------
    Log-likelihood
    """
    
    nobs = args_sampler['nobs']
    locs_m1, stds_m1 = hyperparams['locs_m1'], hyperparams['stds_m1']
    locs_m2, stds_m2 = hyperparams['locs_m2'], hyperparams['stds_m2']
    corr_m1m2 = hyperparams['corr_m1m2']
    norm_m1m2 = hyperparams['norm_m1m2']
    locs_sz, stds_sz = hyperparams['locs_sz'], hyperparams['stds_sz']
    norm_sz = hyperparams['norm_sz']
    var_means, var_covs = functions.get_m1m2_spinz_params(locs_m1, stds_m1, locs_m2, stds_m2, corr_m1m2, locs_sz, stds_sz)
    gwts = hyperparams['gwts']
    kappa = hyperparams['kappa']
    rate = hyperparams['rate']
    injections = args_sampler['injections']
    data = args_sampler['data']
    ngauss = args_sampler['ngauss']
    if np.isscalar(kappa):
        kappa = np.array([kappa] * ngauss)
    
    vt = 0
    sum_dNdz, max_dNdz = 0, 0
    for obsrun in injections.keys():
        nobs_obsrun = len(data[obsrun]['breaks']) - 1
        vt_obsrun, sum_dNdz_obsrun, max_dNdz_obsrun = \
                get_vt_sfr_onepluszm1m2(injections[obsrun], \
                var_means, var_covs, kappa, gwts, norm_m1m2, norm_sz)
        
        vt += vt_obsrun
        sum_dNdz += sum_dNdz_obsrun
        max_dNdz = max(max_dNdz, max_dNdz_obsrun)

    fff = 4
    if 'vfac' in args_sampler.keys():
        vt += args_sampler['vfac'] * vt_obsrun
        fff = 3

    neff = sum_dNdz / max_dNdz
    if neff < fff * nobs:
        return -np.inf
    log_select = nobs * np.log(vt)
    mu = rate * vt
    log_poisson = nobs * np.log(mu) - mu

    logsum_sumprob = 0
    log_sample_size = 0
    for obsrun in data.keys():
        breaks = data[obsrun]['breaks']
        nobs_obsrun = len(breaks) - 1
        parametric_data = data[obsrun]['parametric_data']
        log1pz = data[obsrun]['log1pz']
        analysis_independent = data[obsrun]['analysis_independent']
        dNdz = 0
        for ii in range(ngauss):
            logprob = core_logpdf(parametric_data, var_means[ii], var_covs[ii], gwts[ii])
            # sz_norm effectively allows modeling using truncated normals
            # sz_norm[ii] ** 2 accounts normalisation for both spin1z and spin2z
            dNdz += np.exp(logprob + kappa[ii] * log1pz) / norm_m1m2[ii] / norm_sz[ii] ** 2
        dNdz *= analysis_independent
        
        for ii in range(nobs_obsrun):
            obs_dNdz = dNdz[breaks[ii] : breaks[ii + 1]]
            logsum_sumprob += np.log(np.sum(obs_dNdz))
            log_sample_size += np.log((breaks[ii + 1] - breaks[ii]))

    logsum_sumprob -= log_sample_size
    logsum_sumprob -= log_select
    logsum_sumprob += log_poisson
    
    return logsum_sumprob

############################  VTs  ####################################

def get_vt_sfr_onepluszm1m2(injections, var_means, var_covs,
                           kappa, gwts, norm_m1m2, norm_sz):
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
    kappa: The power-law exponent for the rate evolution
    gwts: Mixing weights for the components
    sz_norm: Normalisation for Gaussians modeling spins
    These are needed as there are no injections outside |sz| > 0.99

    Returns
    -------
    Sensitive volume and the effective number of injection in each redshift 
        segment for the given hyper-parameters.
    """
    
    ngauss = len(var_means)
    var_rec = injections['var_rec']
    w_rec = injections['w_rec']
    log1pz = injections['log1pz']
    ndraw = injections['ndraw']
    
    dNdz = 0
    for ii in range(ngauss):
        logpout = core_logpdf(var_rec, var_means[ii], var_covs[ii], gwts[ii])
        dNdz += np.exp(logpout + kappa[ii] * log1pz) / norm_m1m2[ii] / norm_sz[ii] ** 2
    dNdz *= injections['analysis_independent']
    dNdz *= injections['analysis_time_yr']
    dNdz /= 1e9
    
    dNdz *= w_rec
    sum_dNdz = np.sum(dNdz)
    VT = sum_dNdz / ndraw
    
    return VT, sum_dNdz, np.max(dNdz)

def get_vt_sfr_onepluszmch_binned(injections, var_means, var_covs, min_q,
                                           alphas_q, kappa, gwts, sz_norm):
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
    kappa: The power-law exponent for the rate evolution
    gwts: Mixing weights for the components
    sz_norm: Normalisation for Gaussians modeling spins
    These are needed as there are no injections outside |sz| > 0.99

    Returns
    -------
    Sensitive volume and the effective number of injection in each redshift 
        segment for the given hyper-parameters.
    """
    
    ngauss = len(var_means)
    var_rec = injections['parametric_data']
    log1pz = injections['log1pz']
    logq_rec = injections['logq']
    ndraw = injections['ndraw']
    
    VT, sum_dNdz = [], []
    maxdNdz = 0
    for jj, _ in enumerate(var_rec):
        dNdz = 0
        for ii in range(ngauss):
            logpdfq = powerlaw_logpdf(logq_rec[jj], min_q[ii], 1., alphas_q[ii])
            logpout = core_logpdf(var_rec[jj], var_means[ii], var_covs[ii], 
                                                              logpdfq, gwts[ii])
            dNdz += np.exp(logpout + kappa[ii] * log1pz[jj]) / sz_norm[ii] ** 2
        dNdz *= injections['analysis_independent'][jj]
        dNdz *= injections['analysis_time_yr']
        dNdz /= 1e9
        
        sumdNdz = np.sum(dNdz)
        sum_dNdz.append(sumdNdz)
        VT.append(sumdNdz / ndraw)
        maxdNdz = max(maxdNdz, np.max(dNdz))
    
    return VT, sum_dNdz, np.max(dNdz)

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