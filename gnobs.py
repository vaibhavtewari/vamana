import numpy as np
from conversions import *
from models import *

def get_sfr_pdf(sfr_model, zs, **kwargs):
    """
    Get the probability density for the rate of events at a redshift 
        assuming standard cosmology and star formation model
    sfr_model: defined in models.py
    zs: redshift at which p(zs) is needed
    z_max: maximum redshift for the analysis
    
    Returns
    -------
    Normalised/un-normalised density
    """
    
    z_max = kwargs.get('z_max')
    normalise = kwargs.get('normalise', True)
    dNdz = z_to_dcovdz(zs) * sfr_model(zs, **kwargs) / (1 + zs)
    if normalise:
        dNdz[zs > z_max] = 0
        norm_sfr = get_dNdz_norm(sfr_model, **kwargs)
        return dNdz / norm_sfr
    
    return dNdz

def get_dNdz_norm(sfr_model, **kwargs):
    
    """
    Get the normalisation for density in the redshift distribution
    sfr_model: defined in models.py
    
    Returns
    -------
    The normalisation 
    """
    z_max = kwargs.get('z_max')
    zax = np.expm1(np.linspace(np.log(1.), np.log(1. + z_max), 3000))
    zax[-1] = z_max #above expression does not always end z_max due to precession
    updf_sfr = sfr_model(zax, **kwargs)
    dNdz = updf_sfr * z_to_dcovdz(zax) / (1 + zax)
    norm_sfr = np.trapz(dNdz, zax)
    
    return norm_sfr

def astro_redshifts(sfr_model, **kwargs):
    """
    Sample the redshifts for sources, with a starformation rate
    sfr_model: defined in models.py
    
    Returns
    -------
    The sampled values
    """
    
    z_max = kwargs.get('z_max')
    nsamples = kwargs.get('nsamples')
    
    
    zax = np.expm1(np.linspace(np.log(1.), np.log(1. + z_max), 3000))
    zax[-1] = z_max
    czax = 0.5 * (zax[1:] + zax[:-1])
    pdfz = get_sfr_pdf(sfr_model, czax,  **kwargs)
    cdfz = np.cumsum(pdfz * np.diff(zax))

    #Inverse sampling returns some spurious values
    rndp = np.random.uniform(0., 1., int(nsamples * 1.1)) 
    zastro = np.interp(rndp, cdfz, czax)
    zastro = zastro[(zastro > 0) & (zastro < z_max)]
    
    return np.resize(zastro, nsamples)
