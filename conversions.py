import numpy as np
from astropy.cosmology import Planck15 as cosmo

redshifts = np.expm1(np.linspace(np.log(1.), np.log(1. + 3.0), 3000))
d_lum = cosmo.luminosity_distance(redshifts).value
dcovdz = cosmo.differential_comoving_volume(redshifts).value

def mch_bns():
    return 1.4/2**.2

def dlum_to_z(dl):
    ''' Get the redshift for a luminosity distance'''
    
    return np.interp(dl, d_lum, redshifts)

def z_to_dlum(z):
    ''' Get the redshift for a luminosity distance'''
    
    return np.interp(z, redshifts, d_lum)

def z_to_dcovdz(z):
    ''' Get the redshift for a luminosity distance'''
    
    return 4 * np.pi * np.interp(z, redshifts, dcovdz)

def get_dLdz(z):
    ''' Return the Jacobian to map from distance-detector frame to redshift-source frame
    '''
    
    dddz = np.abs(z_to_dlum(z + 0.0005) - z_to_dlum(z - 0.0005))/0.001
    
    return dddz

def m1m2_to_mcheta(m1, m2):
    '''
    Get chirp mass and symmetric mass ratio from component masses
    '''    
    return (m1*m2)**.6/(m1+m2)**.2, m1*m2/(m1+m2)**2

def m1m2_to_mchq(m1, m2):
    '''
    Get chirp mass and mass ratio from component masses
    '''
    
    if np.isscalar(m1):
        m1 = np.array([m1])
        m2 = np.array([m2])
    mch = (m1*m2)**.6/(m1+m2)**.2
    q = m2/m1
    q[q > 1.] = 1./ q[q > 1.]
    return mch, q

def mcheta_to_m1m2(mch, eta):
    '''
    Get component masses from chirp mass and mass ratio
    '''    
    if eta >= 0.25:
        m1, m2 = mch*2**.2, mch*2**.2
    else:
        a, b = mch**2/eta**.2, mch/eta**.6
        m2 = (b - np.sqrt(b**2 - 4*a))/2
        m1 = a/m2
        
    return m1, m2

def qmch_to_m1m2(mch, q):
    '''
    Get component masses from mass ratio and chirp mass
    '''    
    return mch*(1 + q)**.2/q**.6, q**.4*mch*(1 + q)**.2

def get_pn1(q, s1z, s2z):
    '''
    Get the chi_pn term(see arXiv:1211.0546)
    '''  
    
    chieff = (s1z + q * s2z) / (1 + q)
    chis = 0.5 * (s1z + s2z)
    eta = q / (1 + q) ** 2
    pn1 = chieff - 76 * eta * chis / 113.
    
    return pn1

def get_chieff(m1, m2, s1z, s2z):
    '''
    Get the effective spin
    ''' 
    
    return (m1 * s1z + m2 * s2z) / (m1 + m2)

def J_m1m2_to_mchq(mass1, mass2):
    '''
    Get Jacobian to transfrom from mass1,mass2 -> mchirp, q
    ''' 
    
    mchirp, q = m1m2_to_mchq(mass1, mass2)
    J = mchirp / mass1 ** 2
    
    return 1/J

def J_mchq_to_m1m2(mchirp, q):
    '''
    Get Jacobian to transfrom from mchirp, q -> mass1,mass2
    ''' 
    
    mass1, mass2 = qmch_to_m1m2(mchirp, q)
    J = mchirp / mass1 ** 2
    
    return J

def J_m1m2_to_Mq(m1, m2):
    '''
    Get Jacobian to transfrom from m1, m2 -> M, q
    ''' 
    
    q = m2 / m1
    M = m1 + m2
    J = M / (1 + q) ** 2
    
    return J