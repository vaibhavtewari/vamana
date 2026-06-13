import numpy as np
import scipy.stats as ss


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_norm_m1m2(mu, cov):
    """
    Analytical normalisation for the constraint m2 < m1.

    Computes the probability that m2 < m1 for a 2D Gaussian with
    mean mu and covariance cov, using the fact that m1 - m2 is
    normally distributed.

    Parameters
    ----------
    mu  : array-like of shape (2,), mean vector [mu_m1, mu_m2]
    cov : np.ndarray of shape (2, 2), covariance matrix

    Returns
    -------
    float
        Probability that m2 < m1.
    """
    mu1, mu2 = mu[0], mu[1]
    c11      = cov[0, 0]
    c12      = cov[0, 1]
    c22      = cov[1, 1]
    var_diff = c11 + c22 - 2 * c12
    std_diff = np.sqrt(np.maximum(var_diff, 1e-10))
    return ss.norm.cdf((mu1 - mu2) / std_diff)


def compute_intervals(pdfs, rate, p_lo=5, p_hi=95):
    """
    Compute rate-weighted percentile intervals across posterior samples.

    Parameters
    ----------
    pdfs  : array-like of shape (npost, nax), PDFs per posterior sample
    rate  : array-like of shape (npost,) or scalar, merger rate per sample
    p_lo  : float, lower percentile. Default 5.
    p_hi  : float, upper percentile. Default 95.

    Returns
    -------
    p_lo_vals : np.ndarray of shape (nax,)
    p50_vals  : np.ndarray of shape (nax,)
    p_hi_vals : np.ndarray of shape (nax,)
    """
    pdfs = np.array(pdfs)
    if np.ndim(rate) > 0:
        weighted = np.transpose(rate * np.transpose(pdfs))
    else:
        weighted = pdfs * rate
    return (np.percentile(weighted, p_lo,  axis=0),
            np.percentile(weighted, 50,    axis=0),
            np.percentile(weighted, p_hi,  axis=0))


def _get_sigma(samples, n):
    """Extract actual sigma_m1, sigma_m2 from sigma_over_mu samples."""
    mu_m1  = samples['mu_m1'][n]
    mu_m2  = samples['mu_m2'][n]
    sig_m1 = samples['sigma_over_mu_m1_pow_beta'][n] * mu_m1
    sig_m2 = samples['sigma_over_mu_m2_pow_beta'][n] * mu_m2
    return mu_m1, mu_m2, sig_m1, sig_m2


# =============================================================================
# MASS DISTRIBUTIONS
# =============================================================================

def compute_mass_pdfs(samples, mass_ax, z_eval=0.5):
    """
    Compute differential merger rate on the m1-m2 plane and marginals,
    plus rate evolution factor from z=0 to z_eval.

    Parameters
    ----------
    samples  : dict, posterior samples from load_samples
    mass_ax  : np.ndarray of shape (nm,), mass axis in solar masses
    z_eval   : float, redshift for rate ratio. Default 0.5.

    Returns
    -------
    dict with keys:
        mean_dRdm1dm2 : np.ndarray (nm, nm), mean dR/dm1dm2 at z=0
        pdfs_m1       : np.ndarray (npost, nm), dR/dm1 per sample
        pdfs_m2       : np.ndarray (npost, nm), dR/dm2 per sample
        pdfs_m        : np.ndarray (npost, nm), dR/dm component per sample
        rate_ratio_m1 : np.ndarray (npost, nm), R(z_eval)/R(0) vs m1
        rate_ratio_m2 : np.ndarray (npost, nm), R(z_eval)/R(0) vs m2
    """
    npost  = len(samples['log_post'])
    ncomp  = samples['weights'].shape[1]
    dm     = mass_ax[1] - mass_ax[0]

    m1m2_xy, m1m2_yx = np.meshgrid(mass_ax, mass_ax)
    XYYX             = np.dstack((m1m2_xy, m1m2_yx))
    invalid_idx      = np.where(m1m2_yx / m1m2_xy > 1)

    mean_dRdm1dm2 = np.zeros((len(mass_ax), len(mass_ax)))
    pdfs_m1       = []
    pdfs_m2       = []
    pdfs_m        = []
    rate_ratio_m1 = []
    rate_ratio_m2 = []

    for n in range(npost):
        mu_m1, mu_m2, sig_m1, sig_m2 = _get_sigma(samples, n)
        corr  = samples['rho_m1m2'][n]
        gwts  = samples['weights'][n]
        rate  = samples['rate'][n]
        k     = samples['kcomp'][n]

        pdf_m1m2    = np.zeros_like(m1m2_xy)
        dRdm1dm2_z  = np.zeros_like(m1m2_xy)

        for jj in range(ncomp):
            mean     = np.array([mu_m1[jj], mu_m2[jj]])
            cov      = np.diag([sig_m1[jj]**2, sig_m2[jj]**2])
            cov[0,1] = cov[1,0] = corr[jj] * sig_m1[jj] * sig_m2[jj]

            norm         = calculate_norm_m1m2(mean, cov)
            pdf          = ss.multivariate_normal.pdf(XYYX, mean=mean, cov=cov)
            pdf[invalid_idx] = 0
            pdf         /= norm
            weighted_pdf = pdf * gwts[jj]

            pdf_m1m2   += weighted_pdf
            dRdm1dm2_z += weighted_pdf * (1 + z_eval) ** k[jj]

        mean_dRdm1dm2 += pdf_m1m2 * rate

        rate_ratio_m1.append(
            np.sum(dRdm1dm2_z, axis=0) / np.maximum(np.sum(pdf_m1m2, axis=0), 1e-30)
        )
        rate_ratio_m2.append(
            np.sum(dRdm1dm2_z, axis=1) / np.maximum(np.sum(pdf_m1m2, axis=1), 1e-30)
        )

        pdfs_m1.append(np.sum(pdf_m1m2, axis=0) * dm)
        pdfs_m2.append(np.sum(pdf_m1m2, axis=1) * dm)
        pdfs_m.append(0.5 * np.sum(pdf_m1m2 + pdf_m1m2.T, axis=0) * dm)

    mean_dRdm1dm2 /= npost  # fixed: was /= n (npost-1) in old code

    rate_ratio_m1 = np.array(rate_ratio_m1)
    rate_ratio_m2 = np.array(rate_ratio_m2)
    rate_ratio_m1[np.isnan(rate_ratio_m1)] = 0
    rate_ratio_m2[np.isnan(rate_ratio_m2)] = 0

    return {
        'mean_dRdm1dm2' : mean_dRdm1dm2,
        'pdfs_m1'       : np.array(pdfs_m1),
        'pdfs_m2'       : np.array(pdfs_m2),
        'pdfs_m'        : np.array(pdfs_m),
        'rate_ratio_m1' : rate_ratio_m1,
        'rate_ratio_m2' : rate_ratio_m2,
        'mass_ax'       : mass_ax,
        'z_eval'        : z_eval,
    }


# =============================================================================
# CHIRP MASS AND MASS RATIO DISTRIBUTIONS
# =============================================================================

def compute_mchq_pdfs(samples, mch_ax, q_ax, z_eval=0.5,
                       conversions=None):
    """
    Compute differential merger rate on the chirp mass - mass ratio plane
    and marginals, plus rate evolution factor from z=0 to z_eval.

    Parameters
    ----------
    samples     : dict, posterior samples from load_samples
    mch_ax      : np.ndarray of shape (nmch,), chirp mass axis
    q_ax        : np.ndarray of shape (nq,), mass ratio axis
    z_eval      : float, redshift for rate ratio. Default 0.5.
    conversions : module with qmch_to_m1m2 and J_m1m2_to_mchq functions

    Returns
    -------
    dict with keys:
        pdfs_mch       : np.ndarray (npost, nmch)
        pdfs_q         : np.ndarray (npost, nq)
        rate_ratio_mch : np.ndarray (npost, nmch)
        rate_ratio_q   : np.ndarray (npost, nq)
    """
    if conversions is None:
        raise ValueError("conversions module must be provided")

    npost = len(samples['log_post'])
    ncomp = samples['weights'].shape[1]

    mchq_xy, mchq_yx = np.meshgrid(mch_ax, q_ax)
    m1m2_xy, m1m2_yx = conversions.qmch_to_m1m2(mchq_xy, mchq_yx)
    J                 = conversions.J_m1m2_to_mchq(m1m2_xy, m1m2_yx)
    dmch              = mch_ax[1] - mch_ax[0]
    dq                = q_ax[1]   - q_ax[0]
    invalid_idx       = m1m2_yx / m1m2_xy > 1

    pdfs_mch       = []
    pdfs_q         = []
    rate_ratio_mch = []
    rate_ratio_q   = []

    for n in range(npost):
        mu_m1, mu_m2, sig_m1, sig_m2 = _get_sigma(samples, n)
        corr  = samples['rho_m1m2'][n]
        gwts  = samples['weights'][n]
        k     = samples['kcomp'][n]

        pdf_mchq   = np.zeros_like(mchq_xy)
        dRdmchq_z  = np.zeros_like(mchq_xy)

        for jj in range(ncomp):
            mean     = np.array([mu_m1[jj], mu_m2[jj]])
            cov      = np.diag([sig_m1[jj]**2, sig_m2[jj]**2])
            cov[0,1] = cov[1,0] = corr[jj] * sig_m1[jj] * sig_m2[jj]

            norm         = calculate_norm_m1m2(mean, cov)
            rv           = ss.multivariate_normal(mean=mean, cov=cov)
            pdf          = rv.pdf(np.dstack((m1m2_xy, m1m2_yx))) * J / norm
            pdf[invalid_idx] = 0
            weighted_pdf = pdf * gwts[jj]

            pdf_mchq   += weighted_pdf
            dRdmchq_z  += weighted_pdf * (1 + z_eval) ** k[jj]

        denom_mch = np.maximum(np.sum(pdf_mchq, axis=0), 1e-30)
        denom_q   = np.maximum(np.sum(pdf_mchq, axis=1), 1e-30)

        rate_ratio_mch.append(np.sum(dRdmchq_z, axis=0) / denom_mch)
        rate_ratio_q.append(  np.sum(dRdmchq_z, axis=1) / denom_q)

        pdfs_mch.append(np.sum(pdf_mchq, axis=0) * dq)
        pdfs_q.append(  np.sum(pdf_mchq, axis=1) * dmch)

    rate_ratio_mch = np.array(rate_ratio_mch)
    rate_ratio_q   = np.array(rate_ratio_q)
    rate_ratio_mch[np.isnan(rate_ratio_mch)] = 0
    rate_ratio_q[np.isnan(rate_ratio_q)]     = 0

    return {
        'pdfs_mch'       : np.array(pdfs_mch),
        'pdfs_q'         : np.array(pdfs_q),
        'rate_ratio_mch' : rate_ratio_mch,
        'rate_ratio_q'   : rate_ratio_q,
        'mch_ax'         : mch_ax,
        'q_ax'           : q_ax,
        'z_eval'         : z_eval,
    }


# =============================================================================
# SPIN DISTRIBUTIONS
# =============================================================================

def compute_spin_pdfs(samples, sz_ax, z_eval=0.5, max_spin=0.99):
    """
    Compute differential merger rate as a function of aligned spin sz,
    and rate evolution factor from z=0 to z_eval.

    Uses truncated normal for spin components, truncated to [-max_spin, max_spin].

    Parameters
    ----------
    samples  : dict, posterior samples from load_samples
    sz_ax    : np.ndarray of shape (nsz,), spin axis
    z_eval   : float, redshift for rate ratio. Default 0.5.
    max_spin : float, spin truncation boundary. Default 0.99.

    Returns
    -------
    dict with keys:
        pdfs_sz       : np.ndarray (npost, nsz)
        rate_ratio_sz : np.ndarray (npost, nsz)
    """
    npost = len(samples['log_post'])
    ncomp = samples['weights'].shape[1]

    pdfs_sz       = []
    rate_ratio_sz = []

    for n in range(npost):
        mu_sz = samples['mu_sz'][n]
        sig_sz = samples['sigma_sz'][n]
        gwts   = samples['weights'][n]
        k      = samples['kcomp'][n]

        pdf_sz    = np.zeros(len(sz_ax))
        dRdsz_z   = np.zeros(len(sz_ax))

        for jj in range(ncomp):
            a   = (-max_spin - mu_sz[jj]) / sig_sz[jj]
            b   = ( max_spin - mu_sz[jj]) / sig_sz[jj]
            pdf = ss.truncnorm.pdf(sz_ax, a=a, b=b,
                                   loc=mu_sz[jj], scale=sig_sz[jj])
            weighted_pdf = pdf * gwts[jj]
            pdf_sz   += weighted_pdf
            dRdsz_z  += weighted_pdf * (1 + z_eval) ** k[jj]

        denom = np.maximum(pdf_sz, 1e-30)
        rate_ratio_sz.append(dRdsz_z / denom)
        pdfs_sz.append(pdf_sz)

    rate_ratio_sz = np.array(rate_ratio_sz)
    rate_ratio_sz[np.isnan(rate_ratio_sz)] = 0

    return {
        'pdfs_sz'       : np.array(pdfs_sz),
        'rate_ratio_sz' : rate_ratio_sz,
        'sz_ax'         : sz_ax,
        'z_eval'        : z_eval,
    }


# =============================================================================
# POSTERIOR PREDICTIVE DISTRIBUTION
# =============================================================================

def compute_ppd(samples, ndraws=160, max_spin=0.99, conversions=None):
    """
    Draw posterior predictive samples of m1, m2, sz by sampling
    from the mixture model for each posterior sample.

    Parameters
    ----------
    samples     : dict, posterior samples from load_samples
    ndraws      : int, total expected draws per posterior sample
                  (actual draws per component = round(ndraws * weight * rate))
    max_spin    : float, spin truncation boundary. Default 0.99.
    conversions : module with m1m2_to_mchq function. Optional.

    Returns
    -------
    dict with keys:
        m1, m2, sz        : np.ndarray, component mass and spin samples
        mch, q            : np.ndarray (if conversions provided)
        abs_sz, m         : np.ndarray, derived quantities
    """
    npost = len(samples['log_post'])
    ncomp = samples['weights'].shape[1]

    ppd = {'m1': [], 'm2': [], 'sz': []}

    for n in range(npost):
        mu_m1, mu_m2, sig_m1, sig_m2 = _get_sigma(samples, n)
        corr   = samples['rho_m1m2'][n]
        mu_sz  = samples['mu_sz'][n]
        sig_sz = samples['sigma_sz'][n]
        gwts   = samples['weights'][n]
        rate   = samples['rate'][n]

        for jj in range(ncomp):
            n_draws = int(ndraws * gwts[jj] * rate + 0.5)
            if n_draws == 0:
                continue

            mean     = np.array([mu_m1[jj], mu_m2[jj]])
            cov      = np.diag([sig_m1[jj]**2, sig_m2[jj]**2])
            cov[0,1] = cov[1,0] = corr[jj] * sig_m1[jj] * sig_m2[jj]

            x       = ss.multivariate_normal(mean=mean, cov=cov).rvs(n_draws)
            m1, m2  = x.T if n_draws > 1 else (x[0:1], x[1:2])

            a  = (-max_spin - mu_sz[jj]) / sig_sz[jj]
            b  = ( max_spin - mu_sz[jj]) / sig_sz[jj]
            sz = ss.truncnorm.rvs(a=a, b=b, loc=mu_sz[jj],
                                  scale=sig_sz[jj], size=n_draws)

            ppd['m1'].append(m1)
            ppd['m2'].append(m2)
            ppd['sz'].append(sz)

    ppd['m1'] = np.concatenate(ppd['m1'])
    ppd['m2'] = np.concatenate(ppd['m2'])
    ppd['sz'] = np.concatenate(ppd['sz'])

    # Apply constraints
    valid = ((ppd['m2'] / ppd['m1'] < 1) &
             (np.minimum(ppd['m1'], ppd['m2']) > 0) &
             (np.abs(ppd['sz']) < max_spin))
    ppd['m1'] = ppd['m1'][valid]
    ppd['m2'] = ppd['m2'][valid]
    ppd['sz'] = ppd['sz'][valid]

    # Derived quantities
    ppd['abs_sz'] = np.abs(ppd['sz'])
    mask = np.random.randint(0, 2, size=ppd['m1'].shape).astype(bool)
    ppd['m'] = np.where(mask, ppd['m1'], ppd['m2'])

    if conversions is not None:
        ppd['mch'], ppd['q'] = conversions.m1m2_to_mchq(ppd['m1'], ppd['m2'])

    return ppd


# =============================================================================
# CONDITIONAL DISTRIBUTIONS FROM PPD
# =============================================================================

def compute_conditional_medians(ppd, keys, xax, xvar, yvar, p_lo=5, p_hi=95):
    """
    For each key in keys, compute the conditional median and percentiles
    of yvar given xvar, binned by xax.

    Parameters
    ----------
    ppd   : dict, posterior predictive samples from compute_ppd
    keys  : list of str, identifiers for each (xvar, yvar) pair
    xax   : dict, keyed by key, each value is a 1D bin edge array
    xvar  : dict, keyed by key, name of x variable in ppd
    yvar  : dict, keyed by key, name of y variable in ppd
    p_lo  : float, lower percentile. Default 5.
    p_hi  : float, upper percentile. Default 95.

    Returns
    -------
    p5, p50, p95 : dict keyed by key, each value is np.ndarray of bin medians
    """
    p5  = {}
    p50 = {}
    p95 = {}

    for key in keys:
        bins   = xax[key]
        xdata  = ppd[xvar[key]]
        ydata  = ppd[yvar[key]]
        p5[key]  = []
        p50[key] = []
        p95[key] = []

        for ii in range(len(bins) - 1):
            idx = np.where((xdata > bins[ii]) & (xdata < bins[ii+1]))[0]
            if len(idx) == 0:
                p5[key].append(np.nan)
                p50[key].append(np.nan)
                p95[key].append(np.nan)
            else:
                p5[key].append( np.percentile(ydata[idx], p_lo))
                p50[key].append(np.mean(       ydata[idx]))
                p95[key].append(np.percentile(ydata[idx], p_hi))

        p5[key]  = np.array(p5[key])
        p50[key] = np.array(p50[key])
        p95[key] = np.array(p95[key])

    return p5, p50, p95
