import numpy as np
import scipy.stats as ss

import functions
from models import *

max_spin = 0.99
############################  Chirp Mass  ####################################

def get_proposal_mchirp(locs, stds, dof_prior, max_dcdf, min_mch, 
                        max_mch, alpha, min_mch_scale = 0.15):
    """ 
    Proposes the locations and scales of Gaussians modeling the chirp mass

    Parameters
    ----------
    locs : Current locations of Gaussians
    stds: Current scales of Gaussians
    dof_prior: Not really prior but a set of values to pick degree of freedom from
    dcdf_mch: \Delta F in eq.7 in arXiv:2006.15047
    min_mch: Minimum chirp mass value that can be proposed
    max_mch: Maximum chirp mass value that can be proposed
    alpha: Slope of the power-law proposal function. This is slope of the 
        reference distribution as described in arXiv:2006.15047, but in practice
        a similar powerlaw can also be used.
    min_mch_scale: Proportionality for second column in Table 2 of arXiv:2006.15047

    Returns
    -------
    prp_locs: Proposed location of the Gaussians
    prp_stds: Proposed scales of the Gaussians
    mh_ratio: The Metropolis-Hastings ratio to obtain uniform-in-log prior
        in locations and uniform prior in scales
    """

    ngauss = len(locs)
    prp_stds = []
    mchlocexp = 1.0
    minratio, maxratio = 0.02, max(min_mch_scale, min_mch_scale * (15./ngauss) ** 0.5)
    
    # Propose Locations
    dcdf = np.random.uniform(0, max_dcdf, ngauss)
    cdfs = powerlaw_cdf(locs, min_mch, max_mch, alpha)
    prp_cdfs = get_proposal_uniform(cdfs, np.array([0., 1.0]), dcdf)
    prp_locs = powerlaw_invcdf(prp_cdfs, min_mch, max_mch, alpha)
    
    mhr = powerlaw_pdf(locs, min_mch, max_mch, alpha) 
    mhr /= powerlaw_pdf(prp_locs, min_mch, max_mch, alpha)
    
    mhr *= prp_locs ** -mchlocexp
    mhr *= locs ** mchlocexp
    
    #Propose Scales
    dof = np.random.choice(dof_prior, ngauss)
    minstds, prp_minstds = minratio * locs, minratio * prp_locs
    maxstds, prp_maxstds = maxratio * locs, maxratio * prp_locs
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
    
    mh_ratio = np.prod(mhr * np.exp(f1 - f2) * (c2 / c1))

    return prp_locs, prp_stds, mh_ratio

############################  Aligned Spin  ####################################

def get_proposal_spinz(locs, stds, dof_prior, sz_pls, max_dcdf):
    """ 
    Proposes the locations and scales of Gaussians modeling the aligned spin

    Parameters
    ----------
    locs : Current locations of Gaussians
    stds: Current scales of Gaussians
    dof_prior: Not really prior but a set of values to pick degree of freedom from
    max_dcdf_mch: \Delta F in eq.7 in arXiv:2006.15047
    sz_pls: Location and scale of the proposal function. This are the attributes 
        of the truncated Gaussians as described in arXiv:2006.15047, but in practice
        a similar Gaussian can be also used.
    min_mch_scale: Proportionality for second column in Table 2 of arXiv:2006.15047

    Returns
    -------
    prp_locs: Proposed location of the Gaussians
    prp_stds: Proposed scales of the Gaussians
    norm_norm: The probability of Gaussians between -max_spin and max_spin
    mh_ratio: The Metropolis-Hastings ratio to obtain uniform-in-log prior
        in locations and uniform prior in scales
    """

    sz_prpfnc_loc, szprpfnc_scl = sz_pls
    ngauss, mh_ratio = len(locs), 1
    prp_stds =[]
    minstd = 0.05
    maxstd = 0.5 * (15./ngauss) ** 0.5
    
    # Propose Locations
    dcdf = np.random.uniform(0, max_dcdf, ngauss)
    rvs = ss.norm(loc = sz_prpfnc_loc, scale = szprpfnc_scl)
    cdfs = rvs.cdf(locs)
    cdf_left, cdf_rite = rvs.cdf(-0.5), rvs.cdf(0.5)
    prp_cdfs = get_proposal_uniform(cdfs, np.array([cdf_left, cdf_rite]), dcdf)
    prp_locs = rvs.ppf(prp_cdfs)
    
    mhr = rvs.logpdf(locs) 
    mhr -= rvs.logpdf(prp_locs)
    
    #Propose Scales
    dof = np.random.choice(dof_prior, ngauss)
    rvs = ss.chi2(df = dof, scale = stds/dof)
    cdf_left, cdf_rite = rvs.cdf(minstd), rvs.cdf(maxstd)
    prp_stds = rvs.ppf(np.random.uniform(cdf_left, cdf_rite))
    c2 = cdf_rite - cdf_left

    prp_rvs = ss.chi2(df = dof, scale = prp_stds/dof)
    f1 = prp_rvs.logpdf(stds)
    f2 = rvs.logpdf(prp_stds)
    mhr += (f1 - f2)
    c1 = prp_rvs.cdf(maxstd) - prp_rvs.cdf(minstd)
    
    mh_ratio = np.prod(np.exp(mhr) * (c2 / c1))
        
    norms = ss.norm.cdf(max_spin, prp_locs, prp_stds) 
    norms -= ss.norm.cdf(-max_spin, prp_locs, prp_stds)

    return np.array(prp_locs), np.array(prp_stds), norms, mh_ratio

############################  Rate and Others ####################################

def get_proposal_uniforminlog(val, val_range, max_dcdf):
    """ 
    Proposes values that follow unifrom-in-log distribution

    Parameters
    ----------
    val: Current value
    val_range: Allowed range of values
    d_val: Maximum step size around current value to propose from

    Returns
    -------
    prp_val: Proposed values
    """
    minval, maxval = val_range
    dcdf = np.random.uniform(0, max_dcdf)
    nrm = (np.log(maxval) - np.log(minval))
    cdf = np.log(val) / nrm
    prp_cdf = get_proposal_uniform(cdf, np.array([0., 1.0]), dcdf)
    prp_val = np.exp(prp_cdf * nrm)
    
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

def assign_zseg_idx(zsegidx, zsegfilled, scl = 0.333333):
    """ 
    Assigns new flags to observations.
    See section 2 in https://arxiv.org/pdf/2012.08839.pdf

    Parameters
    ----------
    zsegidx: Current flags for all the observations
    zsegfilled: Array that stores what flags can be assigned to
        individual observations.
    scl: Scale of the Gaussian to make proposals

    Returns
    -------
    prp_zsegidx: New flags
    """
    prp_zsegidx = np.array([]).astype(int)
    #Following is hard coded but determines the fraction of observations
    #that will be considered for new flags. Changing all the flags at once
    #makes the acceptance ratio substantially low
    fsel = np.random.uniform(0.6, 1.0)
    for ii, selidx in enumerate(zsegfilled):
        if np.random.random() > fsel:
            prpzsegidx = int(round(ss.norm.rvs(loc = zsegidx[ii], scale = scl)))
            if prpzsegidx > selidx[-1]:
                prpzsegidx = selidx[0]
            if prpzsegidx < selidx[0]:
                prpzsegidx = selidx[-1]
        else:
            prpzsegidx = zsegidx[ii]
        prp_zsegidx = np.append(prp_zsegidx, prpzsegidx)

    return prp_zsegidx