import numpy as np
from vamana.models import core_logpdf


def get_vt(injections, means, covs, kappa, gwts, norm_m1m2, norm_sz):
    """
    Calculate the sensitive volume-time integral for a single observation
    run, used for correction of selection effects and merger rate calculation.

    Computes the Monte Carlo estimate of the sensitive volume by summing
    population-weighted injection recovery probabilities over all detected
    injections. The effective number of injections is also returned to
    allow a quality check on the Monte Carlo estimate.

    Parameters
    ----------
    injections : dict
        Curated injection data for a single observation run, as stored in
        Data.curated[label].injections. Must contain:
        - 'params_rec' : np.ndarray of shape (ninj, 4),
            recovered parameters [mass1, mass2, spin1z, spin2z]
        - 'w_rec' : np.ndarray of shape (ninj,), injection weights
        - 'log1pz' : np.ndarray of shape (ninj,), log(1 + z) per injection
        - 'ndraw' : float, total number of injections drawn
        - 'analysis_independent' : np.ndarray of shape (ninj,),
            precomputed z_to_dcovdz(z) / (1 + z) / rec_pdf
        - 'analysis_time_yr' : float, total analysis time in years
    means : list of np.ndarray
        Mean vectors of shape (4,) for each mixture component, as returned
        by get_m1m2_spinz_params.
    covs : list of np.ndarray
        Covariance matrices of shape (4, 4) for each mixture component, as
        returned by get_m1m2_spinz_params.
    kappa : np.ndarray
        Power-law exponent for rate evolution, one value per component.
    gwts : np.ndarray
        Mixture weights, one value per component. Must sum to 1.
    norm_m1m2 : np.ndarray
        Normalisation constants for mass Gaussians, one per component.
        Accounts for truncation of the mass distribution.
    norm_sz : np.ndarray
        Normalisation constants for spin Gaussians, one per component.
        Squared in the calculation to account for both spin1z and spin2z.

    Returns
    -------
    VT : float
        Sensitive volume-time integral for this observation run [Gpc^3 yr].
    sum_dNdz : float
        Sum of population-weighted injection recovery probabilities.
        Used as the numerator in the Monte Carlo VT estimate.
    max_dNdz : float
        Maximum population-weighted injection recovery probability.
        Used to compute the effective number of injections:
        neff = sum_dNdz / max_dNdz.

    Notes
    -----
    The VT is divided by 1e9 to convert from Mpc^3 yr to Gpc^3 yr,
    consistent with merger rates quoted in Gpc^-3 yr^-1.

    A quality check on the effective number of injections should be
    performed in the likelihood:
        neff = sum_dNdz / max_dNdz
        if neff < 4 * nobs: return -np.inf

    References
    ----------
    .. [1] Tiwari, V. (2018), "Estimation of the sensitive volume for
           gravitational-wave source populations using weighted Monte Carlo
           integration", Classical and Quantum Gravity,
           https://iopscience.iop.org/article/10.1088/1361-6382/aac89d

    Examples
    --------
    >>> VT, sum_dNdz, max_dNdz = get_vt(
    ...     injections, means, covs,
    ...     kappa, gwts, norm_m1m2, norm_sz
    ... )
    >>> neff = sum_dNdz / max_dNdz
    >>> print(f"VT = {VT:.4f} Gpc^3 yr, neff = {neff:.1f}")
    """
    ngauss  = len(means)
    params_rec = injections['params_rec']
    w_rec   = injections['w_rec']
    log1pz  = injections['log1pz']
    ndraw   = injections['ndraw']

    dNdz = 0
    for ii in range(ngauss):
        logpout = core_logpdf(params_rec, means[ii], covs[ii], gwts[ii])
        dNdz += (
            np.exp(logpout + kappa[ii] * log1pz)
            / norm_m1m2[ii]
            / norm_sz[ii] ** 2
        )

    dNdz *= injections['analysis_independent']
    dNdz *= injections['analysis_time_yr']
    dNdz /= 1e9
    dNdz *= w_rec

    sum_dNdz = np.sum(dNdz)
    VT       = sum_dNdz / ndraw

    return VT, sum_dNdz, np.max(dNdz)