import numpy as np
import scipy.stats as ss

from vamana.models import calculate_norm_m1m2

def mixture_builder(theta, slices, fixed_params):
    """
    Translates sampler values into a structured model payload.
    Uses atleast_1d for population parameters and calculates physical sigmas.
    """

    def get_mixture_params(p):
        """Internal helper to structure components and calculate norms."""
        n_comp = len(np.atleast_1d(p['weights']))
        means, covs = [], []
        norm_m1m2, norm_sz = [], []

        for ii in range(n_comp):
            # 1. Scale sigmas using the Beta-law from prior specs
            mu_m1, mu_m2 = p['mu_m1'][ii], p['mu_m2'][ii]
            mu_sz = p['mu_sz'][ii]
            beta = p['beta']

            sig_m1 = p['sigma_over_mu_m1_pow_beta'][ii] * (mu_m1**beta)
            sig_m2 = p['sigma_over_mu_m2_pow_beta'][ii] * (mu_m2**beta)
            sig_sz = p['sigma_sz'][ii]

            # 2. Structure 4D mean and covariance
            # Mean: [m1, m2, sz1, sz2]
            mean = np.array([mu_m1, mu_m2, mu_sz, mu_sz])
            
            cov = np.diag([sig_m1**2, sig_m2**2, sig_sz**2, sig_sz**2])
            rho_term = sig_m1 * sig_m2 * p['rho_m1m2'][ii]
            cov[0, 1] = cov[1, 0] = rho_term
            
            means.append(mean)
            covs.append(cov)

            # 3. Mass Normalization (m1 - m2 > 0)
            n_m = calculate_norm_m1m2(mean[0:2], cov[0:2, 0:2])
            norm_m1m2.append(n_m)

            # 4. Spin Normalization (sz in [-1, 1])
            n_s = ss.norm.cdf(1, loc=mu_sz, scale=sig_sz) - \
                  ss.norm.cdf(-1, loc=mu_sz, scale=sig_sz)
            norm_sz.append(n_s)

        return means, covs, np.array(norm_m1m2), np.array(norm_sz)

    # --- 1. Unpack & Integrate ---
    p = {name: (theta[idx] if isinstance(idx, list) else theta[idx])
         for name, idx in slices.items()}
    p.update(fixed_params)

    # --- 2. Build Model Components ---
    means, covs, n_m1m2, n_sz = get_mixture_params(p)

    # --- 3. Return Payload ---
    return {
        'means': means,
        'covs': covs,
        'weights': p['weights'],
        'kappa': np.atleast_1d(p['kcomp']), # Mapping prior name kpop to likelihood name kappa
        'rate': p['rate'],
        'norm_m1m2': n_m1m2,
        'norm_sz': n_sz
    }