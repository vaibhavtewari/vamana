import numpy as np
import scipy.stats as ss

import functions
from models import *

max_spin = 0.99
############################  Chirp Mass  ####################################

def get_proposal_mass_locs(locs_m1, locs_m2, max_dcdf, min_mass, max_mass, alpha):
    '''
    Proposes the locations of Gaussians modeling the masses
    
    Parameters
    ----------
    locs_m1 : Current locations of Gaussians modelling primary mass
    locs_m2 : Current locations of Gaussians modelling seondary mass
    max_dcdf: Delta F in eq.7 in arXiv:2006.15047
    min_mass: Minimum mass location that can be proposed
    max_mass: Maximum mass location that can be proposed
    alpha: Slope of the power-law proposal function. This is slope of the 
        reference distribution as described in arXiv:2006.15047, but in practice
        a similar powerlaw can also be used.

    Returns
    -------
    prp_locs: Proposed location of the Gaussians
    mh_ratio: The Metropolis-Hastings ratio to obtain uniform-in-log prior
        in locations and uniform prior in scales
    '''

    ngauss = len(locs_m1)
    
    # Propose Locations
    dcdf = np.random.uniform(0, max_dcdf, ngauss)
    cdfs = powerlaw_cdf(locs_m1, min_mass, max_mass, alpha)
    prp_cdfs = get_proposal_uniform(cdfs, np.array([0., 1.0]), dcdf)
    prp_locs_m1 = powerlaw_invcdf(prp_cdfs, min_mass, max_mass, alpha)
    
    mhr = powerlaw_pdf(locs_m1, min_mass, max_mass, alpha) 
    mhr /= powerlaw_pdf(prp_locs_m1, min_mass, max_mass, alpha)
    
    # We want p(m_1) \propto m_1 and m2 uniform between min_mass and m_1
    # This gives a uniform triangular distribution
    min_m1 = np.maximum(min_mass, 0.15 * locs_m1)
    prp_min_m1 = np.maximum(min_mass, 0.15 * prp_locs_m1)
    mhr *= (prp_locs_m1 - prp_min_m1)
    mhr /= (locs_m1 - min_m1)
    mhr = np.prod(mhr)
    
    #propose m_2
    slope = (prp_locs_m1 - prp_min_m1) / (locs_m1 - min_m1)
    scaled_locs_m2 = prp_min_m1 + (locs_m2 - min_m1) * slope
    prp_locs_m2 = []
    for ii in range(ngauss):
        dcdf = np.random.uniform(0, max_dcdf)
        cdf = powerlaw_cdf(scaled_locs_m2[ii], prp_min_m1[ii], prp_locs_m1[ii], alpha)
        prp_cdf = get_proposal_uniform(cdf, np.array([0., 1.0]), dcdf)
        prp_loc_m2 = powerlaw_invcdf(prp_cdf, prp_min_m1[ii], prp_locs_m1[ii], alpha)
    
        mhr *= powerlaw_pdf(scaled_locs_m2[ii], prp_min_m1[ii], prp_locs_m1[ii], alpha) 
        mhr /= powerlaw_pdf(prp_loc_m2, prp_min_m1[ii], prp_locs_m1[ii], alpha)
        prp_locs_m2 = np.append(prp_locs_m2, prp_loc_m2)
        
    return prp_locs_m1, prp_locs_m2, mhr
        

def get_proposal_mass_stds(locs, prp_locs, stds, dof_prior, min_m_scale, max_m_scale, lexp):
    '''
    Proposes the scales of Gaussians modeling the primary or secondary mass
    
    Parameters
    ----------
    locs : Current locations of Gaussians
    prp_locs : Proposed locations of Gaussians
    stds: Current scales of Gaussians
    dof_prior: Not really prior but a set of values to pick degree of freedom from
    min_m_scale: Minimum proportionality for second column in Table 2 of arXiv:2006.15047
    max_m_scale: Maximum proportionality for second column in Table 2 of arXiv:2006.15047

    Returns
    -------
    prp_stds: Proposed scales of the Gaussians
    mh_ratio: The Metropolis-Hastings ratio to obtain uniform-in-log prior
        in locations and uniform prior in scales
    '''

    ngauss = len(locs)
    minratio = min_m_scale * (15./ngauss) ** 0.5
    maxratio = max_m_scale * (15./ngauss) ** 0.5
    
    #Propose Scales
    dof = np.random.choice(dof_prior, ngauss)
    minstds, prp_minstds = minratio * locs ** lexp, minratio * prp_locs ** lexp
    maxstds, prp_maxstds = maxratio * locs ** lexp, maxratio * prp_locs ** lexp
    slope = (prp_maxstds - prp_minstds) / (maxstds - minstds)
    scaled_stds = prp_minstds + (stds - minstds) * slope

    rvs = ss.chi2(df = dof, scale = scaled_stds/dof)
    cdf_left, cdf_rite = rvs.cdf(prp_minstds), rvs.cdf(prp_maxstds)
    prp_stds = rvs.ppf(np.random.uniform(cdf_left, cdf_rite))
    c2 = cdf_rite - cdf_left

    prp_rvs = ss.chi2(df = dof, scale = prp_stds/dof)
    f1 = prp_rvs.logpdf(scaled_stds)
    f2 = rvs.logpdf(prp_stds)
    c1 = prp_rvs.cdf(prp_maxstds) - prp_rvs.cdf(prp_minstds)
    
    mh_ratio = np.prod(np.exp(f1 - f2) * (c2 / c1))

    return prp_stds, mh_ratio

############################  Aligned Spin  ####################################

def get_proposal_spinz(prp_locs, stds, dof_prior, max_dcdf):
    '''
    Proposes the scales of Gaussians modeling the aligned spin

    Parameters
    ----------
    stds: Current scales of Gaussians
    dof_prior: Not really prior but a set of values to pick degree of freedom from

    Returns
    -------
    prp_stds: Proposed scales of the Gaussians
    norm_norm: The probability of Gaussians between -max_spin and max_spin
    mh_ratio: The Metropolis-Hastings ratio to obtain uniform-in-log prior
        in locations and uniform prior in scales
    '''

    ngauss = len(prp_locs)
    minstd, maxstd = 0.1, 1.3
    
    #Propose Scales
    dof = np.random.choice(dof_prior, ngauss)
    maxstd /= np.sqrt(ngauss)    

    rvs = ss.chi2(df = dof, scale = stds/dof)
    cdf_left, cdf_rite = rvs.cdf(minstd), rvs.cdf(maxstd)
    prp_stds = rvs.ppf(np.random.uniform(cdf_left, cdf_rite))
    c2 = cdf_rite - cdf_left

    prp_rvs = ss.chi2(df = dof, scale = prp_stds/dof)
    f1 = prp_rvs.logpdf(stds)
    f2 = rvs.logpdf(prp_stds)
    c1 = prp_rvs.cdf(maxstd) - prp_rvs.cdf(minstd)

    mh_ratio = np.prod(np.exp(f1 - f2) * (c2 / c1))
        
    norms = ss.norm.cdf(max_spin, prp_locs, prp_stds) 
    norms -= ss.norm.cdf(-max_spin, prp_locs, prp_stds)

    return np.array(prp_stds), norms, mh_ratio

############################  Rate and Others ####################################

def get_proposal_uniforminlog(val, val_range, max_dcdf):
    """ 
    Proposes values that follow unifrom-in-log distribution

    Parameters
    ----------
    val: Current value
    val_range: Allowed range of values
    max_dcdf: Maximum step size around current CDF value to propose from

    Returns
    -------
    prp_val: Proposed values
    """
    minval, maxval = val_range
    log_minval = np.log(minval)
    dcdf = np.random.uniform(0, max_dcdf)
    nrm = (np.log(maxval) - log_minval)
    cdf = (np.log(val) - log_minval) / nrm
    prp_cdf = get_proposal_uniform(cdf, np.array([0., 1.0]), dcdf)
    prp_val = np.exp(prp_cdf * nrm + log_minval)
    
    return prp_val

def get_proposal_uniform(val, val_range, d_val):
    """ 
    Proposes values that follow unifrom distribution.
    Applies reflection boundary condition to proposals outside range.

    Parameters
    ----------
    val: Current value
    val_range: Allowed range of values
    d_val: Maximum step size around current value to propose from

    Returns
    -------
    prp_val: Proposed values
    """
        
    if len(np.array(val_range).shape) == 1:
        minlim, maxlim = val_range
    else:
        minlim, maxlim = val_range[0], val_range[1]

    prp_val = np.random.uniform(val - d_val, val + d_val)
    sflp_val = prp_val * np.sign(prp_val - minlim)
    prp_val = sflp_val + minlim * (1 - np.sign(prp_val - minlim))
    sflp_val = prp_val * np.sign(maxlim - prp_val)
    prp_val = sflp_val + maxlim * (1 - np.sign(maxlim - prp_val))
        
    return prp_val

def get_proposal_gauss(val, loc, scale, d_loc):
    """ 
    Proposes values that follow normal distribution

    Parameters
    ----------
    val: Current value
    loc: Location of Gaussian
    scale: Scale of Gaussian
    d_loc: Maximum step size around current value to propose from

    Returns
    -------
    prp_val: Proposed values
    mh_ratio: Metropolis-Hastings ratio to get the desired distribution
    """
    
    prp_val = np.random.uniform(val - d_loc, val + d_loc)
    mhr = ss.norm.pdf(prp_val, loc = loc, scale = scale)
    mhr /= ss.norm.pdf(val, loc = loc, scale = scale)
    
    return prp_val, np.prod(mhr)

############################  Mixing Fractions  ####################################

def get_mixing_fractions(gwts, dof_prior, ngauss):
    """ 
    Proposes mixing weights that follow uniform prior

    Parameters
    ----------
    gwts: Current mixing weights
    dof_prior: Not really prior but a set of values to pick degree of freedom from
    ngauss: Number of Gaussian

    Returns
    -------
    prp_gwts: Proposed weights
    mh_ratio: Metropolis-Hastings ratio to get the desired distribution
    """
    
    dir_dof = np.random.choice(dof_prior)
    prp_gwts = ss.dirichlet.rvs(alpha = dir_dof * ngauss * gwts  + 1)[0]
    mhr_ratio = ss.dirichlet.pdf(gwts, alpha = dir_dof * ngauss * prp_gwts + 1)
    mhr_ratio /= ss.dirichlet.pdf(prp_gwts, alpha = dir_dof * ngauss * gwts + 1)
        
    return prp_gwts, mhr_ratio

############################  Redshift  ####################################

def assign_zseg_idx(zsegidx, zsegfilled, obs_zseg_split, zseg_nchange):
    """ 
    Assigns new flags to observations.
    See section 2 in https://arxiv.org/pdf/2012.08839.pdf

    Parameters
    ----------
    zsegidx: Current flags for all the observations
    zsegfilled: Array that stores what flags can be assigned to
        individual observations.
    obs_zseg_split: Observations that get split due to redshift segments
    zseg_nchange = number of observation that change redshift segment idx

    Returns
    -------
    prp_zsegidx: New flags
    """
    prp_zsegidx = list(zsegidx)
    obsidx = np.random.choice(obs_zseg_split, zseg_nchange)
    for idx in obsidx:
        prp_zsegidx[idx] += 1
        selidx = zsegfilled[idx]
        if prp_zsegidx[idx] > selidx[-1]:
            prp_zsegidx[idx] = selidx[0]

    return np.array(prp_zsegidx)