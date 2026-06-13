import numpy as np


def get_m1m2_spinz_params(locs_m1, stds_m1, locs_m2, stds_m2, corr_m1m2, locs_sz, stds_sz):
    """
    Organise 1D Gaussians for each mixture component into a 4D Gaussian
    over (mass1, mass2, spin1z, spin2z).

    Parameters
    ----------
    locs_m1 : array-like of float
        Location (mean) of Gaussians modelling primary mass, one per component.
    stds_m1 : array-like of float
        Scale (std) of Gaussians modelling primary mass, one per component.
    locs_m2 : array-like of float
        Location (mean) of Gaussians modelling secondary mass, one per component.
    stds_m2 : array-like of float
        Scale (std) of Gaussians modelling secondary mass, one per component.
    corr_m1m2 : array-like of float
        Correlation coefficient between primary and secondary mass, one per component.
    locs_sz : array-like of float
        Location (mean) of Gaussians modelling aligned spin, one per component.
        Same location is used for both spin1z and spin2z.
    stds_sz : array-like of float
        Scale (std) of Gaussians modelling aligned spin, one per component.
        Same scale is used for both spin1z and spin2z.

    Returns
    -------
    means : list of np.ndarray
        List of mean vectors of shape (4,), one per component.
        Order is [mu_m1, mu_m2, mu_sz, mu_sz].
    covs : list of np.ndarray
        List of covariance matrices of shape (4, 4), one per component.
        Off-diagonal m1-m2 correlation is included.

    Examples
    --------
    >>> means, covs = get_m1m2_spinz_params(
    ...     locs_m1=[30.0], stds_m1=[5.0],
    ...     locs_m2=[20.0], stds_m2=[4.0],
    ...     corr_m1m2=[0.3],
    ...     locs_sz=[0.1], stds_sz=[0.2]
    ... )
    >>> means[0]
    array([30.,  20.,  0.1,  0.1])
    >>> covs[0].shape
    (4, 4)
    """
    ngauss = len(locs_m1)
    means, covs = [], []

    for ii in range(ngauss):
        mean = np.array([locs_m1[ii], locs_m2[ii], locs_sz[ii], locs_sz[ii]])
        cov  = np.diag([stds_m1[ii] ** 2, stds_m2[ii] ** 2,
                        stds_sz[ii] ** 2, stds_sz[ii] ** 2])
        cov[0][1] = cov[1][0] = stds_m1[ii] * stds_m2[ii] * corr_m1m2[ii]
        means.append(mean)
        covs.append(cov)

    return means, covs


def multivariate_normal_logpdf(x, mean, cov):
    """
    Evaluate the log PDF of a multivariate normal distribution.

    Uses eigendecomposition of the covariance matrix for numerical
    stability. See https://gregorygundersen.com/blog/2019/10/30/scipy-multivariate/

    Parameters
    ----------
    x : np.ndarray of shape (nsamples, ndim)
        Data points at which to evaluate the log PDF.
    mean : np.ndarray of shape (ndim,)
        Mean vector of the distribution.
    cov : np.ndarray of shape (ndim, ndim)
        Covariance matrix of the distribution. Must be positive definite.

    Returns
    -------
    np.ndarray of shape (nsamples,)
        Log PDF evaluated at each data point.

    Examples
    --------
    >>> x = np.random.randn(100, 4)
    >>> mean = np.zeros(4)
    >>> cov = np.eye(4)
    >>> logpdf = multivariate_normal_logpdf(x, mean, cov)
    >>> logpdf.shape
    (100,)
    """
    vals, vecs = np.linalg.eigh(cov)
    logdet     = np.sum(np.log(vals))
    valsinv    = 1.0 / vals
    U          = vecs * np.sqrt(valsinv)
    rank       = len(vals)
    dev        = x - mean
    maha       = np.square(np.dot(dev, U)).sum(axis=1)
    log2pi     = np.log(2 * np.pi)
    return -0.5 * (rank * log2pi + maha + logdet)


def core_logpdf(x, mean, cov, gwts):
    """
    Evaluate the weighted log PDF of a single mixture component.

    Parameters
    ----------
    x : np.ndarray of shape (nsamples, ndim)
        Data points at which to evaluate the log PDF.
    mean : np.ndarray of shape (ndim,)
        Mean vector of the mixture component.
    cov : np.ndarray of shape (ndim, ndim)
        Covariance matrix of the mixture component.
    gwts : float
        Mixture weight for this component.

    Returns
    -------
    np.ndarray of shape (nsamples,)
        Weighted log PDF evaluated at each data point.

    Examples
    --------
    >>> logpdf = core_logpdf(x, mean, cov, gwts=0.5)
    >>> logpdf.shape
    (100,)
    """
    return multivariate_normal_logpdf(x, mean, cov) + np.log(gwts)

# in models.py
def calculate_norm_m1m2(mu, cov):
    """
    Analytical normalisation for the constraint m2 < m1.

    Computes the probability that m2 < m1 for a 2D Gaussian with
    mean mu and covariance cov, using the fact that m1 - m2 is
    normally distributed.

    Parameters
    ----------
    mu : array-like of shape (2,)
        Mean vector [mu_m1, mu_m2].
    cov : np.ndarray of shape (2, 2)
        Covariance matrix of the 2D Gaussian.

    Returns
    -------
    float
        Probability that m2 < m1.

    Examples
    --------
    >>> mu  = np.array([30.0, 20.0])
    >>> cov = np.array([[25.0, 5.0], [5.0, 16.0]])
    >>> calculate_norm_m1m2(mu, cov)
    0.987...
    """
    from scipy import stats as ss
    mu1, mu2 = mu[0], mu[1]
    c11      = cov[0, 0]
    c12      = cov[0, 1]
    c22      = cov[1, 1]
    var_diff = c11 + c22 - 2 * c12
    std_diff = np.sqrt(np.maximum(var_diff, 1e-10))
    return ss.norm.cdf((mu1 - mu2) / std_diff)