import glob
import h5py
import numpy as np
from scipy.stats import norm, truncnorm
import gnobs
from models import *
from conversions import *

max_spin = 0.99
z_max = 2.0

def post_process(args_sampler, args_ppd, hyperparams):
    """ 
    Returns posterior predictive for hyper-parameters and that has 
    power-law redshift evolution

    Parameters
    ----------
    args_sampler: Read from the analysis file
    args_ppd: Read from the analysis file
    hyperparams: Population hyper-parameters

    Returns
    -------
    Posterior-predictive
    """
    
    np.random.seed()
    ngauss = args_sampler['ngauss']
    locs_mch = hyperparams['locs_mch']
    stds_mch = hyperparams['stds_mch']
    locs_sz = hyperparams['locs_sz']
    stds_sz = hyperparams['stds_sz']
    min_q = hyperparams['min_q']
    alphas_q = hyperparams['alphas_q']
    kappa = hyperparams['kappa']
    gwts = hyperparams['gwts']
    rate = hyperparams['rate']
    if np.isscalar(kappa):
        kappa = np.array([kappa] * ngauss)
    
    mass_ax = args_ppd['mass_ax']
    q_ax = args_ppd['q_ax']
    sz_ax = args_ppd['sz_ax']
    mch_ax = args_ppd['mch_ax']
    rndn = np.random.randint(ngauss)
    nppd = args_ppd['nppd_per_posterior'] * gwts * 1.1
    nppd = (nppd + 0.5).astype(int)
    
    mass_xy, mass_yx = np.meshgrid(mass_ax, mass_ax)
    mass_xy_flat, mass_yx_flat = mass_xy.flatten(), mass_yx.flatten()
    mch_ax_flat, q_ax_flat = m1m2_to_mchq(mass_xy_flat, mass_yx_flat)
    J = J_mchq_to_m1m2(mch_ax_flat, q_ax_flat)
    
    ppd_mch, ppd_q, ppd_s1z, ppd_s2z, ppd_z = [], [], [], [], []
    post_pdf_mch, post_pdf_sz, post_pdf_q, post_pdf_mchq = 0, 0, 0, 0
    for jj in range(ngauss):
        rvs = norm(loc = locs_mch[jj], scale = stds_mch[jj])
        ppd_mch = np.append(ppd_mch, rvs.rvs(nppd[jj]))
        post_pdf_mch += rvs.pdf(mch_ax) * gwts[jj]
        post_pdf_mchq += rvs.pdf(mch_ax_flat) * powerlaw_pdf(q_ax_flat, min_q[jj], 1., alphas_q[jj]) * gwts[jj]
        
        a, b = (-max_spin - locs_sz[jj]) / stds_sz[jj], (max_spin - locs_sz[jj]) / stds_sz[jj]
        rvs = truncnorm(a = a, b = b, loc = locs_sz[jj], scale = stds_sz[jj])
        ppd_s1z = np.append(ppd_s1z, rvs.rvs(nppd[jj]))
        ppd_s2z = np.append(ppd_s2z, rvs.rvs(nppd[jj]))
        post_pdf_sz += rvs.pdf(sz_ax) * gwts[jj]

        ppd_q = np.append(ppd_q, powerlaw_samples(min_q[jj], 1., alphas_q[jj], nppd[jj]))
        post_pdf_q += powerlaw_pdf(q_ax, min_q[jj], 1., alphas_q[jj]) * gwts[jj]
        ppd_chieff = (ppd_s1z + ppd_q * ppd_s2z) / (1 + ppd_q)
        ppd_z = np.append(ppd_z, gnobs.astro_redshifts(sfr_1pz, nsamples = nppd[jj], z_max = z_max, kappa = kappa[jj]))
            
    post_pdf_m1m2 = J * post_pdf_mchq
        
    post_pdf_mass = []
    for jj, val in enumerate(mass_ax):
        post_pdf_mass = np.append(post_pdf_mass, np.sum(post_pdf_m1m2[jj::len(mass_ax)]))
    post_pdf_mass *= (mass_ax[1] - mass_ax[0])
    
    triu = np.triu(post_pdf_m1m2.reshape(len(mass_ax), len(mass_ax)))
    margm1 = np.sum(triu, axis = 0)
    margm2 = np.sum(triu, axis = 1)
    pm1 = margm1 * (mass_ax[1] - mass_ax[0])
    pm2 = margm2 * (mass_ax[1] - mass_ax[0])
        
    bound = np.sign(ppd_mch - mch_ax[0])
    bound += np.sign(max_spin - np.abs(ppd_s1z))
    bound += np.sign(max_spin - np.abs(ppd_s2z))
    idxsel = np.where(bound == 3)
    np.random.shuffle(idxsel[0])
    
    posterior = {}
    posterior['ppd_mch'] = ppd_mch[idxsel][:args_ppd['nppd_per_posterior']]
    posterior['ppd_chieff'] = ppd_chieff[idxsel][:args_ppd['nppd_per_posterior']]
    posterior['ppd_s1z'] = ppd_s1z[idxsel][:args_ppd['nppd_per_posterior']]
    posterior['ppd_s2z'] = ppd_s2z[idxsel][:args_ppd['nppd_per_posterior']]
    posterior['ppd_q'] = ppd_q[idxsel][:args_ppd['nppd_per_posterior']]
    posterior['ppd_z'] = ppd_z[idxsel][:args_ppd['nppd_per_posterior']]
    
    posterior['post_pdf_mch'] = post_pdf_mch
    posterior['post_pdf_mass'] = 0.5 * np.array(post_pdf_mass)
    posterior['post_pdf_sz'] = post_pdf_sz
    posterior['post_pdf_q'] = post_pdf_q
    posterior['post_pdf_mass1'] = pm1
    posterior['post_pdf_mass2'] = pm2
    
    return posterior

def gather_files(outfile, analysis_name, args_sampler, args_ppd, post_files):
    """ 
    Combines posteriors(saved in individual files) into one

    Parameters
    ----------
    outfile: File name to save combined results
    args_sampler: Read from the analysis file
    args_ppd: Read from the analysis file
    post_files: Posterior files
    """

    posterior, ppd = {}, {}
    for kk, ff in enumerate(post_files):
        with h5py.File(ff, 'r') as filedata:
        
            for key in filedata['posteriors'].keys():
                if key == 'log_lkl' or key == 'rate' or key == 'margl':
                    if kk == 0:
                        posterior[key] = filedata['posteriors'][key][()]
                    else:
                        posterior[key] = np.hstack([posterior[key], filedata['posteriors'][key]])
                else:
                    if kk == 0:
                        posterior[key] = filedata['posteriors'][key][()]
                    else:
                        posterior[key] = np.vstack([posterior[key], filedata['posteriors'][key]])
        
            for key in filedata['ppd'].keys():
                if kk == 0:
                    ppd[key] = filedata['ppd'][key][:]
                else:
                    ppd[key] = np.vstack([ppd[key], filedata['ppd'][key]])
        
    with h5py.File(outfile, 'w') as out:
        
        group = out.create_group('args_sampler')
        for key in args_sampler.keys():
            try:
                group.create_dataset(key, data = args_sampler[key])
            except:
                pass
        group = out.create_group('args_ppd')
        for key in args_ppd.keys():
            try:
                group.create_dataset(key, data = args_ppd[key])
            except:
                pass
        group = out.create_group('posteriors')
        for key in posterior.keys():
            group.create_dataset(key, data = posterior[key])
        group = out.create_group('ppd')
        for key in ppd.keys():
            group.create_dataset(key, data = ppd[key])
            
def get_DiffRate_intervals(pdfs, rates):
    """ 
    Combines rate and density measurement to return credible 
    intervals for differnetial rate

    Parameters
    ----------
    pdfs: Posterior density
    rate: Merger rate
    """
    
    dRd = rates * np.array(pdfs).T
    p5 = np.percentile(dRd, 5., axis = 1)
    p25 = np.percentile(dRd, 25., axis = 1)
    p50 = np.percentile(dRd, 50., axis = 1)
    p75 = np.percentile(dRd, 75., axis = 1)
    p95 = np.percentile(dRd, 95., axis = 1)
    
    return p5, p25, p50, p75, p95