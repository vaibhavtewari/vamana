import numpy as np
from vamana.models import core_logpdf

def calculate_log_likelihood(payload, data, nobs, selection_fnc):
    """
    Likelihood with a user-specifiable selection function.
    """
    # 1. UNPACK PAYLOAD
    means     = payload['means']      
    covs      = payload['covs']       
    weights   = payload['weights']    
    kappa     = payload['kappa']      
    rate      = payload['rate']       
    norm_m1m2 = payload['norm_m1m2'] 
    norm_sz   = payload['norm_sz']

    ncomp = len(weights)

    # --- 2. SELECTION EFFECTS (VT) ---
    vt = 0
    sum_dNdz = 0
    max_dNdz = 0

    # The user-specified selection_fn is called here
    for obsrun in list(data.curated.keys()):
        vt_obs, sum_dN_obs, max_dN_obs = selection_fnc(
            data.curated[obsrun].injections, means, covs, kappa, weights, norm_m1m2, norm_sz
        )
        vt += vt_obs
        sum_dNdz += sum_dN_obs
        max_dNdz = max(max_dNdz, max_dN_obs)

    # Reliability check: neff > 4 * nobs
    if vt > 0:
        neff = sum_dNdz / max_dNdz
        if neff < 4 * nobs:
            return neff
        
        mu = rate * vt
        log_poisson = nobs * np.log(mu) - mu
    else:
        return -np.inf

    # --- 3. PER-EVENT LIKELIHOOD ---
    log_L_events = 0

    for obsrun in list(data.curated.keys()):
        breaks    = data.curated[obsrun].pe['breaks']
        p_data    = data.curated[obsrun].pe['parametric_data']
        log1pz    = data.curated[obsrun].pe['log1pz']
        ai_factor = data.curated[obsrun].pe['analysis_independent']

        total_density = np.zeros(len(p_data))
        for ii in range(ncomp):
            # core_logpdf handles the vectorized Gaussian calculation
            logp = core_logpdf(p_data, means[ii], covs[ii], weights[ii])
            total_density += (
                np.exp(logp + kappa[ii] * log1pz) 
                / norm_m1m2[ii] 
                / (norm_sz[ii] ** 2)
            )
        
        weighted_density = total_density * ai_factor

        for i in range(len(breaks) - 1):
            event_samples = weighted_density[breaks[i]: breaks[i+1]]
            avg_density = np.mean(event_samples)
            
            log_L_events += np.log(avg_density)

    return log_L_events + log_poisson - nobs * np.log(vt)