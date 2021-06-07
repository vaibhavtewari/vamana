import numpy as np
import scipy.stats as ss

import functions

max_spin = 0.99
############################  Chirp Mass  ####################################

def get_proposal_mchirp(
    locs, stds, dof_prior, max_dcdf_mch, min_mch, max_mch, 
                            alpha, min_mch_scale = 0.15):
    """ 
    Proposes the locations and scales of Gaussians modeling the chirp mass

    Parameters
    ----------
    locs : Current locations of Gaussians
    stds: Current scales of Gaussians
    dof_prior: Not really prior but a set of values to pick degree of freedom from
    max_dcdf_mch: \Delta F in eq.7 in arXiv:2006.15047
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

    ngauss, mh_ratio = len(locs), 1
    prp_locs, prp_stds = [], []
    minratio, maxratio, mchlocexp = 0.025, max(min_mch_scale, min_mch_scale * (15./ngauss) ** 0.5), 1.0
    aa = np.random.uniform(alpha - 0.5, alpha + 0.5)
    dof = np.random.choice(dof_prior, ngauss)
    
    for jj in range(ngauss):
        
        loc, std = locs[jj], stds[jj]
        dcdf = np.random.uniform(0, max_dcdf_mch)
        prp_loc, dmch_now = functions.powerlaw_proposal(loc, min_mch, max_mch, aa, dcdf)
        minstd, prp_minstd = minratio * loc, minratio * prp_loc
        maxstd, prp_maxstd = maxratio * loc, maxratio * prp_loc
        pin = loc ** -mchlocexp
        pout = prp_loc ** -mchlocexp
        mh_ratio *= (pout / pin)
        _, dmch_prp = functions.powerlaw_proposal(prp_loc, min_mch, max_mch, aa, dcdf)
        mh_ratio *= (dmch_now/dmch_prp)#Larger dmch_now implies lower probability of finding prp_loc

        rv = ss.chi2(df = dof[jj], scale = std/dof[jj])
        cdf_left, cdf_rite = rv.cdf(prp_minstd), rv.cdf(prp_maxstd)
        prp_std = rv.ppf(np.random.uniform(cdf_left, cdf_rite))

        prp_rv = ss.chi2(df = dof[jj], scale = prp_std/dof[jj])
        f1 = prp_rv.pdf(std)
        f2 = rv.pdf(prp_std)
        mh_ratio *= (f1 / f2)
        c1 = prp_rv.cdf(maxstd) - prp_rv.cdf(minstd)
        c2 = cdf_rite - cdf_left
        mh_ratio *= (c2 / c1)

        prp_locs = np.append(prp_locs, prp_loc)
        prp_stds = np.append(prp_stds, prp_std)

    return prp_locs, prp_stds, mh_ratio

############################  Mass Ratio  ####################################

def get_proposal_q_pl(
        qmins, alphas, qmin_prior, alphaq_prior, d_qmin, d_alphaq):
    """ 
    Proposes the slope and minimum q for the powerlaw modeling mass ratio
        Both priors are uniform

    Parameters
    ----------
    qmins : Fifth column in Table 2 of arXiv:2006.15047
    alphas: Sixth column in Table 2
    qmin_prior: Prior on the minimum mass-ratio(see eq.3)
    alphaq_prior: Slope of the powerlaws modeling the mass-ratio
    d_qmin: Maximum step size around qmin to propose from
    d_alphaq: Step size for alphaq

    Returns
    -------
    prp_qmins: Proposed minimum mass-ratio
    prp_alphas: Proposed alphas
    """
    
    ngauss = len(qmins)
    prp_qmins, prp_alphas = [], []
    for ii in range(ngauss):
        
        dprp = np.random.uniform(0.0, d_qmin)
        prp_qmin = np.random.normal(loc = qmins[ii], scale = dprp)
        
        if prp_qmin < qmin_prior[0]:
            prp_qmin = qmin_prior[0] + (qmin_prior[0] - prp_qmin)
        if prp_qmin > qmin_prior[1]:
            prp_qmin = qmin_prior[1] - (prp_qmin - qmin_prior[1])
        
        dprp = np.random.uniform(0., d_alphaq)
        prp_alpha = np.random.normal(loc = alphas[ii], scale = dprp)
        if prp_alpha < alphaq_prior[0]:
            prp_alpha = alphaq_prior[0] + (alphaq_prior[0] - prp_alpha)
        if prp_alpha > alphaq_prior[1]:
            prp_alpha = alphaq_prior[1] - (prp_alpha - alphaq_prior[1])
        
        prp_alphas = np.append(prp_alphas, prp_alpha)
        prp_qmins = np.append(prp_qmins, prp_qmin)
        
    return prp_qmins, prp_alphas

############################  Aligned Spin  ####################################

def get_proposal_spinz(locs, stds, dof_prior, sz_pls, max_dcdf_sz):
    """ 
    Proposes the locations and scales of Gaussians modeling the aligned spin

    Parameters
    ----------
    locs : Current locations of Gaussians
    stds: Current scales of Gaussians
    dof_prior: Not really prior but a set of values to pick degree of freedom from
    max_dcdf_mch: \Delta F in eq.7 in arXiv:2006.15047
    sz_pls: Location and scale of the proposal function. This are the attribtes 
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
    prp_locs, prp_stds =[], []
    minstd = prp_minstd = 0.05
    maxstd = prp_maxstd = 0.25 * (15./ngauss) ** 0.5
    norm_norm = []

    for jj in range(ngauss):
        
        loc, std = locs[jj], stds[jj]
        dcdf = np.random.uniform(0, max_dcdf_sz)
        prp_loc, dsz_now = functions.gauss_proposal(loc, sz_prpfnc_loc, szprpfnc_scl, dcdf)
        
        bound = 0
        while bound < 2:
            dof = np.random.choice(dof_prior)
            rv = ss.chi2(df = dof, scale = std/dof)
            prp_std = rv.rvs()
            bound = np.sign(prp_std - prp_minstd) + np.sign(prp_maxstd - prp_std)
            
        prp_stds.append(prp_std)
        prp_rv = ss.chi2(df = dof, scale = prp_std/dof)
        f1 = prp_rv.pdf(std)
        f2 = rv.pdf(prp_std)
        mh_ratio *= (f1 / f2)
        c1 = prp_rv.cdf(maxstd) - prp_rv.cdf(minstd)
        c2 = rv.cdf(prp_maxstd) - rv.cdf(prp_minstd)
        mh_ratio *= (c2 / c1)

        _, dsz_prp = functions.gauss_proposal(prp_loc, sz_prpfnc_loc, szprpfnc_scl, dcdf)
        mh_ratio *= (dsz_now/dsz_prp)
        prp_locs.append(prp_loc)
        
        norm = ss.norm.cdf(max_spin, prp_loc, prp_std) 
        norm -= ss.norm.cdf(-max_spin, prp_loc, prp_std)
        norm_norm = np.append(norm_norm, norm)

    return np.array(prp_locs), np.array(prp_stds), norm_norm, mh_ratio

############################  Rate and Others ####################################

def get_proposal_uniforminlog(val, val_range, d_val):
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
    mh_ratio: Metropolis-Hastings ratio to get the desired distribution
    """
    
    dprp = np.random.uniform(0.0, d_val)
    prp_val = np.random.normal(loc = val, scale = dprp)
        
    if prp_val < val_range[0]:
        prp_val = val_range[0] + (val_range[0] - prp_val)
    if prp_val > val_range[1]:
        prp_val = val_range[1] - (prp_val - val_range[1])
    
    mh_ratio = val/prp_val
    
    return prp_val, mh_ratio

def get_proposal_uniform(val, val_range, d_val):
    """ 
    Proposes values that follow unifrom distribution

    Parameters
    ----------
    val: Current value
    val_range: Allowed range of values
    d_val: Maximum step size around current value to propose from

    Returns
    -------
    prp_val: Proposed values
    """
    
    dprp = np.random.uniform(0.0, d_val)
    prp_val = np.random.normal(loc = val, scale = dprp)
        
    if prp_val < val_range[0]:
        prp_val = val_range[0] + (val_range[0] - prp_val)
    if prp_val > val_range[1]:
        prp_val = val_range[1] - (prp_val - val_range[1])
    
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
    
    return prp_val, mhr

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