import numpy as np
import scipy.stats as ss
from scipy.stats import beta as beta_dist
from scipy.stats import dirichlet as dirichlet_dist
from scipy.interpolate import interp1d


# =============================================================================
# INVERSE CDF (PRIOR TRANSFORM) FUNCTIONS
# Map u ~ Uniform[0,1] -> theta (physical space).
# =============================================================================

def uniform(u, lo, hi, *args, **kwargs):
    return lo + u * (hi - lo)


def uniform_in_log(u, lo, hi, *args, **kwargs):
    return lo * (hi / lo) ** u


def fixed(u, val, *args, **kwargs):
    return val


def m1_for_uniform_m1m2_ordered(u, lo, hi, *args, **kwargs):
    """
    Inverse CDF for mu_m1 using order statistics to ensure
    mu_m1_0 < mu_m1_1 < ... < mu_m1_{ncomp-1}.
    """
    theta       = kwargs['theta']
    param_names = kwargs['param_names']
    name        = kwargs['name']
    comp_idx    = int(name.split('_')[-1])
    ncomp       = len([n for n in param_names if n.startswith('mu_m1_')])
    u_ordered   = beta_dist.ppf(u, comp_idx + 1, ncomp - comp_idx)
    return lo + u_ordered * (hi - lo)


def m1_for_uniform_m1m2(u, lo, hi, *args, **kwargs):
    """
    Inverse CDF for mu_m1. Uses sqrt(u) for joint uniformity in m1-m2 plane.
    """
    return lo + (u**0.5) * (hi - lo)


def uniform_min_m2_to_m1(u, lo, *args, **kwargs):
    """
    Inverse CDF for mu_m2, uniformly distributed between lo and mu_m1_i.
    """
    theta       = kwargs['theta']
    param_names = kwargs['param_names']
    name        = kwargs['name']
    comp_idx    = name.split('_')[-1]
    m1_idx      = param_names.index(f'mu_m1_{comp_idx}')
    m1_val      = theta[m1_idx]
    return lo + u * (m1_val - lo)


def dirichlet(u, *args, **kwargs):
    """
    Inverse CDF for Dirichlet-distributed mixture weights via stick-breaking.
    """
    theta       = kwargs['theta']
    param_names = kwargs['param_names']
    name        = kwargs['name']
    parts       = name.split('_')
    base_name   = '_'.join(parts[:-1])
    comp_idx    = int(parts[-1])
    indices     = [i for i, n in enumerate(param_names)
                   if n.startswith(f"{base_name}_")]
    K           = len(indices)
    used_weight = np.sum([theta[indices[i]] for i in range(comp_idx)])
    remaining   = max(1.0 - used_weight, 0.0)
    if comp_idx == K - 1:
        return remaining
    exponent = 1.0 / (K - comp_idx - 1)
    return remaining * (1.0 - u ** exponent)


def piecewise_uniform_logarithmic(u, a, b, **kwargs):
    """
    Inverse CDF of a piecewise distribution:
    uniform on [-a, a], proportional to 1/|x| on [-b, -a] and [a, b].
    """
    u   = np.atleast_1d(u)
    c1  = 1.0 / (2 * a * (1 + np.log(b / a)))
    u_l = 0.5 - c1 * a
    u_r = 0.5 + c1 * a
    x   = np.zeros_like(u)
    m_l = u < u_l
    m_m = (u >= u_l) & (u <= u_r)
    m_r = u > u_r
    x[m_l] = -a * np.exp((u_l - u[m_l]) / (a * c1))
    x[m_m] = (u[m_m] - 0.5) / c1
    x[m_r] =  a * np.exp((u[m_r] - u_r) / (a * c1))
    return x[0] if x.size == 1 else x


def uniform_kcomp(u, half_width=2.0, *args, **kwargs):
    """
    Inverse CDF for kcomp, uniformly distributed around kpop.
    """
    theta       = kwargs['theta']
    param_names = kwargs['param_names']
    kpop_idx    = param_names.index('kpop')
    kpop        = theta[kpop_idx]
    lo          = kpop - half_width
    hi          = kpop + half_width
    return lo + u * (hi - lo)


# =============================================================================
# CDF AND LOG-PDF HELPERS
# Used internally by proposal functions. Not used by prior_transform.
# Only defined here for distributions whose CDF is non-trivial.
# uniform and uniform_in_log are trivially invertible and inlined directly
# in their proposal functions.
# =============================================================================

def _cdf_piecewise_uniform_logarithmic(x, a, b):
    """
    CDF of the piecewise uniform-logarithmic distribution.

    Companion to piecewise_uniform_logarithmic (the inverse CDF).
    Used by piecewise_uniform_log_draw to step in CDF space.

    Parameters
    ----------
    x : float
        Value in physical space.
    a : float
        Half-width of the uniform central region.
    b : float
        Outer boundary of the logarithmic tails.

    Returns
    -------
    float
        CDF value in [0, 1].
    """
    c1  = 1.0 / (2 * a * (1 + np.log(b / a)))
    u_l = 0.5 - c1 * a
    u_r = 0.5 + c1 * a
    if   x <= -b: return 0.0
    elif x <  -a: return float(u_l - c1 * a * np.log(-x / a))
    elif x <=  a: return float(0.5 + c1 * x)
    elif x <   b: return float(u_r + c1 * a * np.log(x / a))
    else:         return 1.0


def _logpdf_piecewise_uniform_logarithmic(x, a, b):
    """
    Log-PDF of the piecewise uniform-logarithmic distribution.

    Used by piecewise_uniform_log_draw to compute the MH correction.
    The distribution is uniform on [-a, a] and proportional to 1/|x|
    on [-b, -a] and [a, b].

    Parameters
    ----------
    x : float
        Value in physical space.
    a : float
        Half-width of the uniform central region.
    b : float
        Outer boundary of the logarithmic tails.

    Returns
    -------
    float
        Log-PDF value.
    """
    c1 = 1.0 / (2 * a * (1 + np.log(b / a)))
    ax = abs(x)
    if   ax <= a: return np.log(c1)
    elif ax <  b: return np.log(c1 * a / ax)
    else:         return -np.inf


# =============================================================================
# DOF POOL GENERATION
# =============================================================================

def get_dof(min_nu, max_nu, nsamp=10000):
    """
    Generate a pool of dof values with roughly constant fractional change.
    """
    ax      = np.linspace(min_nu, max_nu, nsamp).astype(int)
    chi_std = []
    for a in ax:
        std  = ss.chi2.ppf(0.5 + 0.3413447, df=a, scale=1.0 / a)
        std -= ss.chi2.ppf(0.5 - 0.3413447, df=a, scale=1.0 / a)
        std /= 2
        chi_std.append(std)
    chi_std  = np.array(chi_std)
    cdf      = np.cumsum(chi_std)
    cdf     /= cdf[-1]
    interp   = interp1d(cdf, ax, bounds_error=False, fill_value='extrapolate')
    dof_pool = interp(np.linspace(min(cdf), 1, nsamp)).astype(int)
    return dof_pool


# =============================================================================
# SINGLE PARAMETER PROPOSAL FUNCTIONS
# Convention: <name>_draw(current, *args, **kwargs) -> (proposed, log_mhc)
#
# The draw function owns its MH correction entirely.
# Dynesty never calls these.
# =============================================================================

def _reflect(val, lo, hi):
    """
    Apply reflective boundary condition to val in [lo, hi].
    Faithful to get_proposal_uniform boundary handling in old code.

    A proposal that overshoots the boundary bounces back rather than
    being clipped — preserving proposal continuity and avoiding pile-up.
    """
    val = float(val)
    # Reflect at lower boundary
    sgn = np.sign(val - lo)
    val = sgn * val + lo * (1 - sgn)
    # Reflect at upper boundary
    sgn = np.sign(hi - val)
    val = sgn * val + hi * (1 - sgn)
    return val

def powerlaw_draw(current, lo, hi, alpha, **kwargs):
    """
    Propose value from power-law on [lo, hi] by stepping in CDF space.
    If hi is None, looks up mu_m1_i from theta for mu_m2 proposals.

    Returns (proposed, log_mhc).
    """
    if hi is None:
        param_names = kwargs['param_names']
        name        = kwargs['name']
        comp_idx    = name.split('_')[-1]
        theta       = kwargs['theta']
        m1_idx      = param_names.index(f'mu_m1_{comp_idx}')
        hi          = float(theta[m1_idx])

    delta_u  = kwargs['delta_u']
    dcdf     = np.random.uniform(0, delta_u)
    beta_pl  = alpha + 1.0

    def _cdf(x):
        return (x**beta_pl - lo**beta_pl) / (hi**beta_pl - lo**beta_pl)

    def _invcdf(p):
        return (p * (hi**beta_pl - lo**beta_pl) + lo**beta_pl) ** (1.0/beta_pl)

    def _log_pdf(x):
        return (np.log(abs(beta_pl))
                - np.log(abs(hi**beta_pl - lo**beta_pl))
                + alpha * np.log(x))

    cdf_proposed = _reflect(
        _cdf(current) + np.random.uniform(-dcdf, dcdf), 1e-6, 1 - 1e-6
    )
    proposed = float(np.clip(_invcdf(cdf_proposed), lo + 1e-6, hi - 1e-6))
    log_mhc  = float(_log_pdf(current) - _log_pdf(proposed))
    return proposed, log_mhc


def chi2_draw(current, lo, hi, **kwargs):
    """
    Propose value from truncated chi2 centred on current, truncated to [lo, hi].

    Returns (proposed, log_mhc).
    """
    dof_pool = kwargs['dof_pool']
    dof      = int(np.random.choice(dof_pool)) if dof_pool is not None else 10

    rv_fwd    = ss.chi2(df=dof, scale=current/dof)
    c_lo      = rv_fwd.cdf(lo)
    c_hi      = rv_fwd.cdf(hi)
    proposed  = float(np.clip(
        rv_fwd.ppf(np.random.uniform(c_lo, c_hi)), lo + 1e-8, hi - 1e-8
    ))
    c2        = c_hi - c_lo

    rv_rev    = ss.chi2(df=dof, scale=proposed/dof)
    f1        = rv_rev.logpdf(current)
    f2        = rv_fwd.logpdf(proposed)
    c1        = rv_rev.cdf(hi) - rv_rev.cdf(lo)

    log_mhc   = float(f1 - f2 + np.log(c2) - np.log(c1))
    return proposed, log_mhc


def uniform_draw(current, lo, hi, **kwargs):
    """
    Propose a new value for a uniformly distributed parameter by stepping
    in CDF space.

    The CDF of Uniform[lo, hi] is u = (x - lo) / (hi - lo), so a step
    of delta_u in CDF space corresponds to a step of delta_u * (hi - lo)
    in theta space. The proposal is symmetric so log_mhc = 0 always.

    Uses the existing inverse CDF function `uniform` for the back-transform.
    This is the proposal counterpart to the inverse CDF function `uniform`.
    Used for parameters declared with distribution 'uniform' in the prior
    file that have no other proposal function defined.

    Parameters
    ----------
    current : float
        Current value of the parameter in physical space.
    lo : float
        Lower bound of the uniform distribution.
    hi : float
        Upper bound of the uniform distribution.
    **kwargs
        delta_u : float, step size in CDF space. Required.

    Returns
    -------
    proposed : float
        Proposed value in physical space.
    log_mhc : float
        Log Metropolis-Hastings correction. 0.0 for uniform since log-PDF is flat.
    """
    delta_u    = kwargs['delta_u']
    dcdf       = np.random.uniform(0, delta_u)
    u_current  = (current - lo) / (hi - lo)
    u_proposed = _reflect(u_current + np.random.uniform(-dcdf, dcdf), 1e-6, 1 - 1e-6)
    return float(uniform(u_proposed, lo, hi)), 0.0


def uniform_in_log_draw(current, lo, hi, **kwargs):
    """
    Propose a new value for a log-uniformly distributed parameter by
    stepping in CDF space.

    The CDF of LogUniform[lo, hi] is u = log(x/lo) / log(hi/lo), so
    equal steps in CDF space correspond to equal fractional steps in
    theta space. The log-PDF is -log(x) - log(log(hi/lo)), so
    log_mhc = log(proposed/current).

    Uses the existing inverse CDF function `uniform_in_log` for the
    back-transform. This is the proposal counterpart to `uniform_in_log`.
    Used for parameters declared with distribution 'uniform_in_log' in
    the prior file that have no other proposal function defined.

    Parameters
    ----------
    current : float
        Current value of the parameter in physical space.
    lo : float
        Lower bound of the log-uniform distribution.
    hi : float
        Upper bound of the log-uniform distribution.
    **kwargs
        delta_u : float, step size in CDF space. Required.
        theta       : np.ndarray, current full parameter vector.
        param_names : list of str, parameter names in theta order.

    Returns
    -------
    proposed : float
        Proposed value in physical space.
    log_mhc : float
        Log Metropolis-Hastings correction = log(proposed/current).
    """
    delta_u    = kwargs['delta_u']
    dcdf       = np.random.uniform(0, delta_u)
    u_current  = np.log(current / lo) / np.log(hi / lo)
    u_proposed = _reflect(u_current + np.random.uniform(-dcdf, dcdf), 1e-6, 1 - 1e-6)
    proposed   = float(uniform_in_log(u_proposed, lo, hi))
    log_mhc    = np.log(proposed / current)
    return proposed, float(log_mhc)


def piecewise_uniform_logarithmic_draw(current, a, b, **kwargs):
    """
    Propose a new value for a piecewise uniform-logarithmic parameter
    by stepping in CDF space.

    Steps by Uniform(-dcdf, dcdf) in CDF space, then maps back via
    the inverse CDF piecewise_uniform_logarithmic. Unlike uniform
    distributions, the PDF is not flat so the MH correction is
    non-zero and must be computed.

    This is the proposal counterpart to the inverse CDF function
    `piecewise_uniform_logarithmic`. Used for parameters declared with
    distribution 'piecewise_uniform_logarithmic' in the prior file
    that have no other proposal function defined (e.g. kpop).

    Parameters
    ----------
    current : float
        Current value of the parameter in physical space.
    a : float
        Half-width of the uniform central region.
    b : float
        Outer boundary of the logarithmic tails.
    **kwargs
        delta_u : float, step size in CDF space. Required.

    Returns
    -------
    proposed : float
        Proposed value in physical space.
    log_mhc : float
        Log Metropolis-Hastings correction =
        log_pdf(current) - log_pdf(proposed).
    """
    delta_u    = kwargs['delta_u']
    dcdf       = np.random.uniform(0, delta_u)
    u_current  = _cdf_piecewise_uniform_logarithmic(current, a, b)
    u_proposed = _reflect(u_current + np.random.uniform(-dcdf, dcdf), 1e-6, 1 - 1e-6)
    proposed   = float(piecewise_uniform_logarithmic(u_proposed, a, b))
    log_mhc    = (_logpdf_piecewise_uniform_logarithmic(current,  a, b)
                - _logpdf_piecewise_uniform_logarithmic(proposed, a, b))
    return proposed, float(log_mhc)


def uniform_kcomp_draw(current, half_width, **kwargs):
    """
    Propose a new value for a per-component rate evolution parameter
    (kcomp) by stepping in CDF space.

    kcomp is uniformly distributed on [kpop - half_width, kpop + half_width],
    where kpop is the global rate evolution parameter read from the current
    theta. Since the bounds are symmetric and the distribution is uniform,
    the proposal is symmetric in CDF space and log_mhc = 0.

    Uses the existing inverse CDF function `uniform_kcomp` for the
    back-transform. This is the proposal counterpart to `uniform_kcomp`.
    Used for parameters declared with distribution 'uniform_kcomp' in
    the prior file.

    Parameters
    ----------
    current : float
        Current value of kcomp in physical space.
    half_width : float
        Half-width of the uniform distribution around kpop.
        Declared as arg1 in the prior file.
    **kwargs
        delta_u     : float, step size in CDF space. Required.
        theta       : np.ndarray, current full parameter vector.
        param_names : list of str, parameter names in theta order.

    Returns
    -------
    proposed : float
        Proposed value in physical space.
    log_mhc : float
        Log Metropolis-Hastings correction. 0.0 since log-PDF is flat.
    """
    theta         = kwargs['theta']          # theta_proposed — has new kpop
    theta_current = kwargs['theta_current']  # theta_current — has old kpop
    param_names   = kwargs['param_names']
    kpop_idx      = param_names.index('kpop')

    # New bounds from proposed kpop
    new_kpop = theta[kpop_idx]
    lo       = new_kpop - half_width
    hi       = new_kpop + half_width

    # Fractional position in old bounds — preserved when kpop moves
    old_kpop  = theta_current[kpop_idx]
    f         = (current - (old_kpop - half_width)) / (2 * half_width)

    delta_u    = kwargs['delta_u']
    dcdf       = np.random.uniform(0, delta_u)
    u_proposed = _reflect(f + np.random.uniform(-dcdf, dcdf), 1e-6, 1 - 1e-6)
    return float(uniform_kcomp(u_proposed, half_width, **kwargs)), 0.0


# =============================================================================
# JOINT PARAMETER PROPOSAL FUNCTIONS
# Convention: <name>_joint_draw(theta_group, dof_pool, **kwargs)
#                 -> (proposed_group, log_mhc)
#
# theta_group and proposed_group are in physical theta space.
# Identified by _joint suffix. Dynesty never calls these.
# =============================================================================

def dirichlet_joint_draw(gwts, dof_pool, **kwargs):
    """
    Propose mixture weights using Dirichlet proposal centred on current weights.
    Returns (proposed_weights, log_mhc).
    """
    ngauss    = len(gwts)
    dof       = np.random.choice(dof_pool)
    alpha_fwd = dof * ngauss * gwts + 1
    prp_gwts  = dirichlet_dist.rvs(alpha=alpha_fwd)[0]
    prp_gwts  = np.clip(prp_gwts, 1e-10, None)
    prp_gwts /= prp_gwts.sum()
    alpha_rev = dof * ngauss * prp_gwts + 1
    log_mhc   = float(dirichlet_dist.logpdf(gwts,    alpha_rev) -
                      dirichlet_dist.logpdf(prp_gwts, alpha_fwd))
    return prp_gwts, log_mhc


def mass_powerlaw_joint_draw(theta_group, dof_pool, **kwargs):
    """
    Jointly propose mu_m1, mu_m2, sigma_m1, sigma_m2 per component.

    Faithful to arXiv:2006.15047. Per-component structure supports
    future physics-inspired correlations.

    theta_group order (prior file order):
        [mu_m1_0,...,mu_m1_{n-1},
         mu_m2_0,...,mu_m2_{n-1},
         sigma_over_mu_m1_pow_beta_0,...,sigma_over_mu_m1_pow_beta_{n-1},
         sigma_over_mu_m2_pow_beta_0,...,sigma_over_mu_m2_pow_beta_{n-1}]
    """

    proposal_args     = kwargs['proposal_args']
    member_args       = kwargs['member_args']
    delta_u           = kwargs['delta_u']
    fixed_params      = kwargs['fixed_params']
    group_param_names = kwargs['group_param_names']
    alpha             = float(proposal_args[0])
    beta_exp  = float(fixed_params['beta'])
    min_ratio = float(fixed_params['min_mu_m1_over_mu_m2'])

    m1_args      = member_args['mu_m1']
    lo           = float(m1_args[0])
    hi           = float(m1_args[1])
    s1_args      = member_args['sigma_over_mu_m1_pow_beta']
    s2_args      = member_args['sigma_over_mu_m2_pow_beta']
    s1_min_ratio = float(s1_args[0])
    s1_max_ratio = float(s1_args[1])
    s2_min_ratio = float(s2_args[0])
    s2_max_ratio = float(s2_args[1])

    # Slice theta_group by parameter name — order independent
    mu_m1_mask = [i for i, n in enumerate(group_param_names)
                  if n.startswith('mu_m1_')]
    mu_m2_mask = [i for i, n in enumerate(group_param_names)
                  if n.startswith('mu_m2_')]
    s1_mask    = [i for i, n in enumerate(group_param_names)
                  if n.startswith('sigma_over_mu_m1')]
    s2_mask    = [i for i, n in enumerate(group_param_names)
                  if n.startswith('sigma_over_mu_m2')]

    mu_m1            = theta_group[mu_m1_mask]
    mu_m2            = theta_group[mu_m2_mask]
    sigma_over_mu_m1 = theta_group[s1_mask]
    sigma_over_mu_m2 = theta_group[s2_mask]
    ncomp            = len(mu_m1)

    sigma_m1 = sigma_over_mu_m1 * mu_m1 ** beta_exp
    sigma_m2 = sigma_over_mu_m2 * mu_m2 ** beta_exp

    beta_pl = alpha + 1.0

    def _cdf(x, lo_i, hi_i):
        return (x**beta_pl - lo_i**beta_pl) / (hi_i**beta_pl - lo_i**beta_pl)

    def _invcdf(p, lo_i, hi_i):
        return (p*(hi_i**beta_pl - lo_i**beta_pl) + lo_i**beta_pl)**(1.0/beta_pl)

    def _log_pdf(x, lo_i, hi_i):
        return (np.log(abs(beta_pl))
                - np.log(abs(hi_i**beta_pl - lo_i**beta_pl))
                + alpha * np.log(x))

    def _propose_sigma_i(sigma_i, mu_i, prp_mu_i, min_ratio, max_ratio):
        """
        Propose sigma for one component using truncated chi2.
        Scales current sigma to proposed mass range before proposing.
        Faithful to get_proposal_mass_stds from arXiv:2006.15047.
        """
        sigma_lo     = min_ratio * mu_i     ** beta_exp
        sigma_hi     = max_ratio * mu_i     ** beta_exp
        prp_sigma_lo = min_ratio * prp_mu_i ** beta_exp
        prp_sigma_hi = max_ratio * prp_mu_i ** beta_exp

        slope        = (prp_sigma_hi - prp_sigma_lo) / (sigma_hi - sigma_lo)
        scaled_sigma = np.clip(
            prp_sigma_lo + (sigma_i - sigma_lo) * slope,
            prp_sigma_lo + 1e-8, prp_sigma_hi - 1e-8
        )

        dof       = int(np.random.choice(dof_pool))
        rv        = ss.chi2(df=dof, scale=scaled_sigma/dof)
        cdf_left  = rv.cdf(prp_sigma_lo)
        cdf_right = rv.cdf(prp_sigma_hi)
        prp_sigma = float(np.clip(
            rv.ppf(np.random.uniform(cdf_left, cdf_right)),
            prp_sigma_lo + 1e-8, prp_sigma_hi - 1e-8
        ))
        c2        = cdf_right - cdf_left

        prp_rv    = ss.chi2(df=dof, scale=prp_sigma/dof)
        f1        = prp_rv.logpdf(scaled_sigma)
        f2        = rv.logpdf(prp_sigma)
        c1        = prp_rv.cdf(prp_sigma_hi) - prp_rv.cdf(prp_sigma_lo)

        return prp_sigma, float(f1 - f2 + np.log(c2) - np.log(c1))

    # --- Propose mu_m1 for all components ---
    # Draw random step size per component: dcdf ~ Uniform(0, delta_u)
    # then step by Uniform(-dcdf, dcdf) — equivalent to triangular distribution
    cdfs_m1     = _cdf(mu_m1, lo, hi)
    dcdf_m1     = np.random.uniform(0, delta_u, ncomp)
    prp_cdfs_m1 = np.array([
        _reflect(cdfs_m1[i] + np.random.uniform(-dcdf_m1[i], dcdf_m1[i]),
                 1e-6, 1 - 1e-6)
        for i in range(ncomp)
    ])
    prp_m1     = np.clip(
        np.array([_invcdf(p, lo, hi) for p in prp_cdfs_m1]),
        lo + 1e-6, hi - 1e-6
    )
    min_m1     = np.maximum(lo, min_ratio * mu_m1)
    prp_min_m1 = np.maximum(lo, min_ratio * prp_m1)

    log_mhc  = np.sum(
        [_log_pdf(mu_m1[i], lo, hi) - _log_pdf(prp_m1[i], lo, hi)
         for i in range(ncomp)]
    )
    log_mhc += np.sum(
        np.log(prp_m1 - prp_min_m1) - np.log(mu_m1 - min_m1)
    )

    # --- Per-component: propose mu_m2, sigma_m1, sigma_m2 ---
    slope        = (prp_m1 - prp_min_m1) / (mu_m1 - min_m1)
    scaled_mu_m2 = prp_min_m1 + (mu_m2 - min_m1) * slope

    prp_m2       = np.zeros(ncomp)
    prp_sigma_m1 = np.zeros(ncomp)
    prp_sigma_m2 = np.zeros(ncomp)

    for i in range(ncomp):
        # Draw random step size: dcdf ~ Uniform(0, delta_u)
        dcdf_m2    = np.random.uniform(0, delta_u)
        cdf_m2     = _cdf(scaled_mu_m2[i], prp_min_m1[i], prp_m1[i])
        prp_cdf_m2 = _reflect(
            cdf_m2 + np.random.uniform(-dcdf_m2, dcdf_m2),
            1e-6, 1 - 1e-6
        )
        prp_m2[i]  = np.clip(
            _invcdf(prp_cdf_m2, prp_min_m1[i], prp_m1[i]),
            prp_min_m1[i] + 1e-6, prp_m1[i] - 1e-6
        )
        log_mhc += (
            _log_pdf(scaled_mu_m2[i], prp_min_m1[i], prp_m1[i]) -
            _log_pdf(prp_m2[i],       prp_min_m1[i], prp_m1[i])
        )

        prp_sigma_m1[i], lmhc_s1 = _propose_sigma_i(
            sigma_m1[i], mu_m1[i], prp_m1[i], s1_min_ratio, s1_max_ratio
        )
        log_mhc += lmhc_s1

        prp_sigma_m2[i], lmhc_s2 = _propose_sigma_i(
            sigma_m2[i], mu_m2[i], prp_m2[i], s2_min_ratio, s2_max_ratio
        )
        log_mhc += lmhc_s2

    prp_sigma_over_mu_m1 = prp_sigma_m1 / prp_m1 ** beta_exp
    prp_sigma_over_mu_m2 = prp_sigma_m2 / prp_m2 ** beta_exp

    # Assemble prp_group in same order as theta_group
    prp_group               = np.zeros(len(theta_group))
    prp_group[mu_m1_mask]   = prp_m1
    prp_group[mu_m2_mask]   = prp_m2
    prp_group[s1_mask]      = prp_sigma_over_mu_m1
    prp_group[s2_mask]      = prp_sigma_over_mu_m2
    return prp_group, float(log_mhc)


def spin_joint_draw(theta_group, dof_pool, **kwargs):
    """
    Jointly propose mu_sz and sigma_sz per component.

    Faithful to get_proposal_spinz from arXiv:2006.15047.

    mu_sz   : step in piecewise_uniform_logarithmic CDF space (analytic)
    sigma_sz: truncated chi2 centred on current sigma_sz

    theta_group order:
        [mu_sz_0,...,mu_sz_{n-1},
         sigma_sz_0,...,sigma_sz_{n-1}]

    No proposal_args needed — delta_u controls step size, dof_pool controls chi2.
    """

    member_args       = kwargs['member_args']
    delta_u           = kwargs['delta_u']
    group_param_names = kwargs['group_param_names']

    mu_args  = member_args['mu_sz']
    sig_args = member_args['sigma_sz']
    mu_a     = float(mu_args[0])
    mu_b     = float(mu_args[1])
    sig_lo   = float(sig_args[0])
    sig_hi   = float(sig_args[1])

    # Slice by parameter name — order independent
    mu_mask  = [i for i, n in enumerate(group_param_names)
                if n.startswith('mu_sz_')]
    sig_mask = [i for i, n in enumerate(group_param_names)
                if n.startswith('sigma_sz_')]

    mu_sz  = theta_group[mu_mask]
    sig_sz = theta_group[sig_mask]
    ncomp  = len(mu_sz)

    # CDF and log-PDF for piecewise_uniform_logarithmic — use shared helpers
    def _pul_cdf(x):
        return _cdf_piecewise_uniform_logarithmic(x, mu_a, mu_b)

    def _pul_log_pdf(x):
        return _logpdf_piecewise_uniform_logarithmic(x, mu_a, mu_b)

    prp_mu  = np.zeros(ncomp)
    prp_sig = np.zeros(ncomp)
    log_mhc = 0.0

    for i in range(ncomp):
        # Draw random step size: dcdf ~ Uniform(0, delta_u)
        dcdf_mu    = np.random.uniform(0, delta_u)
        cdf_mu     = _pul_cdf(mu_sz[i])
        prp_cdf_mu = _reflect(
            cdf_mu + np.random.uniform(-dcdf_mu, dcdf_mu),
            1e-6, 1 - 1e-6
        )
        prp_mu[i]  = float(piecewise_uniform_logarithmic(prp_cdf_mu, mu_a, mu_b))
        # no log_mhc contribution for mu_sz — CDF step uses prior function, symmetric

        # Propose sigma_sz_i — truncated chi2 centred on current sigma_sz
        dof       = int(np.random.choice(dof_pool))
        rv        = ss.chi2(df=dof, scale=sig_sz[i]/dof)
        cdf_left  = rv.cdf(sig_lo)
        cdf_right = rv.cdf(sig_hi)
        prp_s     = float(np.clip(
            rv.ppf(np.random.uniform(cdf_left, cdf_right)),
            sig_lo + 1e-8, sig_hi - 1e-8
        ))
        c2        = cdf_right - cdf_left

        prp_rv    = ss.chi2(df=dof, scale=prp_s/dof)
        f1        = prp_rv.logpdf(sig_sz[i])
        f2        = rv.logpdf(prp_s)
        c1_s      = prp_rv.cdf(sig_hi) - prp_rv.cdf(sig_lo)

        prp_sig[i] = prp_s
        log_mhc   += float(f1 - f2 + np.log(c2) - np.log(c1_s))

    # Assemble prp_group in same order as theta_group
    prp_group            = np.zeros(len(theta_group))
    prp_group[mu_mask]   = prp_mu
    prp_group[sig_mask]  = prp_sig
    return prp_group, float(log_mhc)