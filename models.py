import sys
import numpy as np
from scipy import interpolate
from scipy.stats import norm
from scipy.integrate import nquad, quad
from scipy.interpolate import interp1d

from math import exp

maxspin = 0.99

# PDF for the two caninical models
def prob_flatlog(m1, m2, **kwargs):
    ''' Density for masses following unifrom-in-log distribution'''

    min_mass1 = kwargs.get('min_mass1', 5)
    max_mass1 = kwargs.get('max_mass1', 50.)
    min_mass2 = kwargs.get('min_mass2', 5.)
    max_mass2 = kwargs.get('max_mass2', 50.)
    m1, m2 = np.array(m1), np.array(m2)
    
    C_flatlog_m1 = 1 / (np.log(max_mass1) - np.log(min_mass1))
    C_flatlog_m2 = 1 / (np.log(max_mass2) - np.log(min_mass2))
    
    p_m1_m2 = C_flatlog_m1 * C_flatlog_m2 * (1./m1) * (1./m2)
    bound = (np.sign(max_mass1 - m1) * np.sign(m1 - min_mass1))
    idx = np.where(bound < 0)
    p_m1_m2[idx] = 0
    bound = (np.sign(max_mass2 - m2) * np.sign(m2 - min_mass2))
    idx = np.where(bound < 0)
    p_m1_m2[idx] = 0    

    return p_m1_m2

def prob_powerlaw(m1, m2, **kwargs): 
    
    min_mass = kwargs.get('min_mass', 5.)
    max_mass = kwargs.get('max_mass', 50.)
    max_mtotal = kwargs.get('max_mtotal', 2 * max_mass)
    alpha = kwargs.get('alpha', 2.35)
  
    m1, m2 = np.array(m1), np.array(m2)    
    
    C_powerlaw = (max_mtotal/2.)**(-alpha + 1)/(-alpha + 1) - min_mass**(-alpha + 1)/(-alpha + 1)    
    C_powerlaw += quad(lambda x: x**(-alpha) * (max_mass - x)/(x - min_mass), max_mtotal/2., max_mass)[0]   
    
    bound = np.sign(max_mtotal - m1 - m2) + np.sign(max_mass - m1) + np.sign(m2 - min_mass) + np.sign(m1 - m2)
    idx = np.where(bound != 4)
    
    p_m1_m2 = np.zeros_like(m1)
    p_m1_m2 = (1. / C_powerlaw) * m1**(-alpha) /(m1 - min_mass)
    p_m1_m2[idx] = 0    
        
    return p_m1_m2


def prob_uniform(m1, m2, **kwargs):

    min_mass1 = kwargs.get('min_mass1', 5.)
    max_mass1 = kwargs.get('max_mass1', 50.)
    min_mass2 = kwargs.get('min_mass2', 5.)
    max_mass2 = kwargs.get('max_mass2', 50.)   
    
    p_m1_m2 = np.ones_like(m1)/(max_mass1 - min_mass1)/(max_mass2 - min_mass2)
    bound = (np.sign(max_mass1 - m1) * np.sign(m1 - min_mass1))
    idx = np.where(bound < 0)
    p_m1_m2[idx] = 0
    bound = (np.sign(max_mass2 - m2) * np.sign(m2 - min_mass2))
    idx = np.where(bound < 0)
    p_m1_m2[idx] = 0  
               
    return p_m1_m2
    
# Functions to generate samples for the two canonical models
def draw_powerlaw_samples(**kwargs):
    '''
    Yields random masses, with the first component drawn from
    the Salpeter initial mass function distribution and the
    second mass drawn uniformly between min_mass and the mass of
    the first component.
    '''
 
    nsamples = kwargs.get('nsamples', 1)
    min_mass = kwargs.get('min_mass', 5.)
    max_mass = kwargs.get('max_mass', 50.)
    max_mtotal = kwargs.get('max_mtotal', 2 * max_mass)
    alpha = kwargs.get('alpha', 2.35)    
        
    a = (max_mass/min_mass)**(-alpha + 1.0) - 1.0
    beta = 1.0 / (-alpha + 1.0)
    
    k = nsamples * int(2.0 + np.log(1 + 100./nsamples))
    aa = min_mass * (1.0 + a * np.random.random(k))**beta
    bb = np.random.uniform(min_mass, aa, k) 
    
    idx = np.where(aa + bb < max_mtotal)
    m1, m2 = (np.maximum(aa, bb))[idx], (np.minimum(aa, bb))[idx]
    
    m1.resize(nsamples)
    m2.resize(nsamples)    
    
    if m1[-1] == 0:
        sys.exit("Rejection sampling including zeros in the population masses!")    
        
    return m1, m2

def draw_flatlog_samples(**kwargs):
    '''
    Yields random masses drawn uniformly-in-log between min_mass
    and max_mass.
    '''
    #PDF doesnt match with sampler
    nsamples = kwargs.get('nsamples', 1)
    min_mass1 = kwargs.get('min_mass1', 5.)
    max_mass1 = kwargs.get('max_mass1', 50.)
    min_mass2 = kwargs.get('min_mass2', 5.)
    max_mass2 = kwargs.get('max_mass2', 50.) 
    
    flatlogmin1 = np.log(min_mass1)
    flatlogmax1 = np.log(max_mass1)
    flatlogmin2 = np.log(min_mass2)
    flatlogmax2 = np.log(max_mass2)
    
    mass1 = np.exp(np.random.uniform(flatlogmin1, flatlogmax1, nsamples))
    mass2 = np.exp(np.random.uniform(flatlogmin2, flatlogmax2, nsamples))
        
    return mass1, mass2

def draw_uniform_samples(**kwargs):
    '''
    Yields random masses drawn uniformly-in-log between min_mass
    and max_mass, discarding those with a total mass exceeding
    max_mtotal.
    '''
    #PDF doesnt match with sampler
    nsamples = kwargs.get('nsamples', 1)
    min_mass_m1 = kwargs.get('min_mass1', 5.)
    max_mass_m1 = kwargs.get('max_mass1', 50.)
    min_mass_m2 = kwargs.get('min_mass2', 5.)
    max_mass_m2 = kwargs.get('max_mass2', 50.)    
    
    m1 = np.random.uniform(min_mass_m1, max_mass_m1, nsamples)
    m2 = np.random.uniform(min_mass_m2, max_mass_m2, nsamples)
        
    return m1, m2

def draw_normal_samples(**kwargs):
    '''
    Yields random masses drawn uniformly-in-log between min_mass
    and max_mass, discarding those with a total mass exceeding
    max_mtotal.
    '''
    #PDF doesnt match with sampler
    nsamples = kwargs.get('nsamples', 1)
    loc_mass = kwargs.get('loc_mass')
    scl_mass = kwargs.get('scl_mass')
    
    m1 = np.random.normal(loc_mass, scl_mass, nsamples)
    m2 = np.random.normal(loc_mass, scl_mass, nsamples)
        
    return m1, m2
    
#Spinz samples
def draw_astro_spinz(**kwargs):
    
    spin_key = kwargs.get('spin_key')
    nsamples = kwargs.get('nsamples', 1)
    max_spin = kwargs.get('max_spin')
    
    if spin_key == 'low_isotropic':
        s1t = np.random.triangular(0, 0, max_spin, nsamples)
        s2t = np.random.triangular(0, 0, max_spin, nsamples)
        
        rndn1 = np.random.uniform(-1, 1, nsamples)
        rndn2 = np.random.uniform(-1, 1, nsamples)        
        
        return s1t*rndn1, s2t*rndn2
    
    if spin_key == 'uniform_z':
        s1z = np.random.uniform(-max_spin, max_spin, nsamples)
        s2z = np.random.uniform(-max_spin, max_spin, nsamples)
        
        return s1z, s2z
       
    if key == 'zero':
        return np.zeros(nsamples)
    
# Functions to get p(sz) for some given aligned spin models
def spinz_models(spinsz, **kwargs):
    ''' Return probability for spins defined by Will Farr et. al.'''
    
    spin_key = kwargs.get('spin_key')
    max_spin = kwargs.get('max_spin', maxspin)
    spinsz = np.array(spinsz)
    
    bound = np.sign(np.absolute(max_spin) - np.absolute(spinsz))
    bound += np.sign(1 - np.absolute(spinsz))	
    idx = np.where(bound != 2) 
    
    if spin_key == 'precessing':
        ''' prob(sz) given isotropic spin distribution'''
        pdf = (np.log(max_spin) - np.log(np.abs(spinzs))) / 2 / max_spin #-- Currently considering only flat isotropic
        pdf[idx] = 0
        
        return pdf
    
    if spin_key == 'aligned':
        ''' Returns the PDF of mass when spins are aligned and isotropic in magnitude'''    
        pdf = (np.ones_like(spinsz) / 2 / max_spin)
        idx = np.where(bound != 2)
        pdf[idx] = 0
        
        return pdf
    
    if spin_key == 'disable_spin':
        ''' Returns unit array '''    
        pdf = np.ones_like(spinsz)
        pdf[idx] = 0
        
        return pdf
    
# Functions to get p(z) for some given star formation models
def sfr_madau_dickinson(zs, **kwargs):
    ''' Return star formation rate -- Madau Dickinson 2014'''
    ''' This is not normalized '''  
    ''' zp = 1.9, alphaz = 2.7, betaz = 2.9 reported in the paper'''
    
    z_max = kwargs.get('z_max')
    zp = kwargs.get('zp')
    alphaz = kwargs.get('alphaz')
    betaz = kwargs.get('betaz')
    
    zs = np.array(zs)
    pdf = (1 + zs) ** alphaz / ( 1 + ((1. + zs)/(1. + zp))**(alphaz + betaz) )
    
    return pdf

def sfr_1pz(zs, **kwargs):
    ''' This is not normalized '''  
    
    z_max = kwargs.get('z_max')
    kappa = kwargs.get('kappa')
    
    zs = np.array(zs)
    pdf = (1 + zs) ** kappa
    
    return pdf

def UinComov(zs):
    ''' Return a unity array as zs
    '''
    
    pdf = np.ones_like(zs)
    
    return pdf

############ 1D samplers and pdf #####################

def powerlaw_samples(minx, maxx, alpha, nsamples):
    '''
    Draw power-law samples
    '''
    a = (maxx/minx)**(-alpha + 1.0) - 1.0
    beta = 1.0 / (-alpha + 1.0)
    
    x = minx * (1.0 + a * np.random.random(nsamples))**beta
    
    return x

def powerlaw_pdf(x, minx, maxx, alpha):
    '''
    Yields power-law pdf for range minx, maxx and exponent alpha
    '''
    
    C = (-alpha + 1.0) / (maxx ** (-alpha + 1.0) - minx ** (-alpha + 1.0))
    
    pdf = C * x ** (-alpha)
    if np.isscalar(pdf):
        if x < minx or x > maxx:
            return 0
        else:
            return pdf
    pdf[(x < minx) | (x > maxx)] = 0
    
    return pdf

def powerlaw_logpdf(logx, minx, maxx, alpha):
    '''
    Yields power-law logpdf for range minx, maxx and exponent alpha
    '''
    
    minlogx, maxlogx = np.log(minx), np.log(maxx)
    C = (-alpha + 1.0) / (maxx ** (-alpha + 1.0) - minx ** (-alpha + 1.0))
    
    logpdf = np.log(C) - alpha * logx
    logpdf[(logx < minlogx) | (logx > maxlogx)] = -np.inf
    
    return logpdf

def powerlaw_cdf(x, minx, maxx, alpha):
    '''
    Yields power-law cdf for range minx, maxx and exponent alpha
    '''
    bound = bound_range(x, minx, maxx)
    if np.min(bound) < 0:
        print ('Points out of provided range!')
        return 0
    
    if alpha == 1:
        C = 1/(np.log(maxx) - np.log(minx))
        cdf = C * (np.log(x) - np.log(minx))
        return cdf
    else:
        B = (maxx ** (-alpha + 1.0) - minx ** (-alpha + 1.0))
        cdf = (x ** (-alpha + 1.0) - minx ** (-alpha + 1.0)) / B
        
    return cdf

def powerlaw_invcdf(cdf, minx, maxx, alpha):
    '''
    Yields inverse CDF(PPF) for range minx, maxx and exponenet alpha
    '''  
    
    if alpha == 1:
        C = 1/(np.log(maxx) - np.log(minx))
        icdf = np.exp(cdf/C + np.log(minx))
    else:
        B = (maxx ** (-alpha + 1.00) - minx ** (-alpha + 1.00))
        icdf = (cdf * B + minx ** (-alpha + 1.00)) ** (1/(-alpha + 1.00))

    return icdf
    
def broken_powerlaw_pdf(x, xrange, alpha):
    '''
    Yields broken, power-law pdf for range minx, maxx and exponent alpha
    '''
    minx, maxx = xrange[:-1], xrange[1:]
    pdf = np.zeros_like(x)
    C = np.zeros_like(minx)
    idx = np.where(alpha != 1)
    C[idx] = (maxx[idx] ** (-alpha[idx] + 1.0) - minx[idx] ** (-alpha[idx] + 1.0)) / (-alpha[idx] + 1.0)
    idx = np.where(-alpha == 1)
    C[idx] = (np.log(maxx[idx]) - np.log(minx[idx]))
    mult = np.cumprod(minx[1:] ** (-alpha[:-1] + alpha[1:]))
    C[1:] *= mult
    
    mult, C = 1, 1./np.sum(C)
    for ii in range(len(minx)):
        idx = np.where((x >= minx[ii]) & (x < maxx[ii]))
        pdf[idx] = C * x[idx] ** -alpha[ii]
        
        if ii > 0:
            mult *= minx[ii] ** (-alpha[ii - 1] + alpha[ii])
            pdf[idx] *= mult
    
    pdf[(x < minx[0]) | (x > maxx[-1])] = 0
    
    return pdf


def sample_broken_powerlaw(xrange, alpha, size):
    '''
    Yields broken, power-law samples for range minx, maxx and exponent alpha
    Uses Inverse sampling instead of more clever analytic calculation
    '''
    
    rndn = np.random.uniform(0, 1., size)
    xsamp = broken_powerlaw_invcdf(rndn, xrange, alpha)
    
    return xsamp

def broken_powerlaw_invcdf(points, xrange, alpha):
    
    ax = np.linspace(xrange[0], xrange[-1], 100000)
    pdf = broken_powerlaw_pdf(ax, xrange, alpha)
    cdf = np.cumsum(pdf * (ax[1] - ax[0]))
    interp = interp1d(cdf, ax, bounds_error = False, fill_value = 'extrapolate')
    icdf = interp(points)
    
    return icdf

def broken_exponential_pdf(x, xrange, alpha):
    '''
    Yields broken, exponential pdf for range minx, maxx and exponent alpha
    '''
    minx, maxx = xrange[:-1], xrange[1:]
    pdf = np.zeros_like(x)
    C = (1./alpha) * (np.exp(alpha * maxx) - np.exp(alpha * minx))
    mult = np.cumprod(np.exp(minx[1:] * (alpha[:-1] - alpha[1:])))
    C[1:] *= mult
    
    mult, C = 1, 1./np.sum(C)
    for ii in range(len(minx)):
        idx = np.where((x >= minx[ii]) & (x < maxx[ii]))
        pdf[idx] = C * np.exp(x[idx] * alpha[ii])
        
        if ii > 0:
            mult *= np.exp(minx[ii] * (-alpha[ii] + alpha[ii - 1]))
            pdf[idx] *= mult
    
    return pdf

def bound_range(x, minx, maxx, p =  5):
    ''' conditional to assess if x lies between minx and maxx'''
    bound = np.sign(np.round(x, p) - np.round(minx, p))
    bound *= np.sign(np.round(maxx, p) - np.round(x, p))
    
    return bound

def core_logpdf(x, mean, cov, gwts):
    multivarnorm_logpdf = multivariate_normal_logpdf(x, mean, cov)
    logpdf = multivarnorm_logpdf + np.log(gwts)
    return logpdf

def multivariate_normal_logpdf(x, mean, cov):
    # https://gregorygundersen.com/blog/2019/10/30/scipy-multivariate/
    # `eigh` assumes the matrix is Hermitian.
    vals, vecs = np.linalg.eigh(cov)
    logdet     = np.sum(np.log(vals))
    valsinv    = 1 / vals
    # `vecs` is R times D while `vals` is a R-vector where R is the matrix 
    # rank. The asterisk performs element-wise multiplication.
    U          = vecs * np.sqrt(valsinv)
    rank       = len(vals)
    dev        = x - mean
    # "maha" for "Mahalanobis distance".
    maha       = np.square(np.dot(dev, U)).sum(axis = 1)
    log2pi     = np.log(2 * np.pi)
    return -0.5 * (rank * log2pi + maha + logdet)

def truncnorm_pdf(x, idxnonzero, ll, rl, loc, scale):
    
    pdf = np.zeros_like(x)
    rv = norm(loc = loc, scale = scale)
    truncpdf = norm.pdf(x[idxnonzero])
    truncpdf /= (rv.cdf(rl) - rv.cdf(ll))
    pdf[idxnonzero] = truncpdf
    
    return pdf
    