import glob
import h5py
import numpy as np
from scipy.stats import norm, multivariate_normal, truncnorm
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
    locs_m1 = hyperparams['locs_m1']
    stds_m1 = hyperparams['stds_m1']
    locs_m2 = hyperparams['locs_m2']
    stds_m2 = hyperparams['stds_m2']
    corr_m1m2 = hyperparams['corr_m1m2']
    norm_m1m2 = hyperparams['norm_m1m2']
    
    locs_sz = hyperparams['locs_sz']
    stds_sz = hyperparams['stds_sz']

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
    
    mass_xy, mass_yx = np.meshgrid(mass_ax, mass_ax)
    pdf_m1m2 = 0
    for jj in range(ngauss):
        mean = np.array([locs_m1[jj], locs_m2[jj]])
        cov = np.diag([stds_m1[jj] ** 2, stds_m2[jj] ** 2])
        cov[0][1] = cov[1][0] = corr_m1m2[jj] * stds_m1[jj] * stds_m2[jj]
        rvs = multivariate_normal(mean = mean, cov = cov)
        pdf = rvs.pdf(np.dstack((mass_xy, mass_yx)))
        pdf[mass_yx / mass_xy > 1] = 0
        nrm = pdf.sum() * (mass_ax[1] - mass_ax[0]) ** 2
        pdf /= nrm
        pdf_m1m2 += pdf * gwts[jj]
        
    post_pdf_m1 = np.sum(pdf_m1m2, axis = 0) * (mass_ax[1] - mass_ax[0])
    post_pdf_m2 = np.sum(pdf_m1m2, axis = 1) * (mass_ax[1] - mass_ax[0])
    post_pdf_m = 0.5 * (np.sum(pdf_m1m2 + pdf_m1m2.T, axis = 0) * (mass_ax[1] - mass_ax[0]))
    
    mchq_xy, mchq_yx = np.meshgrid(mch_ax, q_ax)
    m1m2_xy, m1m2_yx = qmch_to_m1m2(mchq_xy, mchq_yx)
    J = J_m1m2_to_mchq(m1m2_xy, m1m2_yx)
    pdf_mchq = 0
    for jj in range(ngauss):
        mean = np.array([locs_m1[jj], locs_m2[jj]])
        cov = np.diag([stds_m1[jj] ** 2, stds_m2[jj] ** 2])
        cov[0][1] = cov[1][0] = corr_m1m2[jj] * stds_m1[jj] * stds_m2[jj]
        rvs = multivariate_normal(mean = mean, cov = cov)
        pdf = rvs.pdf(np.dstack((m1m2_xy, m1m2_yx))) * J / norm_m1m2[jj]
        pdf[m1m2_yx / m1m2_xy > 1] = 0
        pdf_mchq += pdf * gwts[jj]
        
    post_pdf_mch = np.sum(pdf_mchq, axis = 0) * (q_ax[1] - q_ax[0])
    post_pdf_q = np.sum(pdf_mchq, axis = 1) * (mch_ax[1] - mch_ax[0])
    
    post_pdf_sz = 0
    for jj in range(ngauss):
        a, b = (-max_spin - locs_sz[jj]) / stds_sz[jj], (max_spin - locs_sz[jj]) / stds_sz[jj]
        rvs = truncnorm(a = a, b = b, loc = locs_sz[jj], scale = stds_sz[jj])
        post_pdf_sz += rvs.pdf(sz_ax) * gwts[jj]
    
    posterior = {}
    
    posterior['post_pdf_mchirp'] = post_pdf_mch
    posterior['post_pdf_mass'] = post_pdf_m
    posterior['post_pdf_sz'] = post_pdf_sz
    posterior['post_pdf_q'] = post_pdf_q
    posterior['post_pdf_mass1'] = post_pdf_m1
    posterior['post_pdf_mass2'] = post_pdf_m2
    
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
            
def get_DiffRate_intervals(pdfs, rates, intervals):
    """ 
    Combines rate and density measurement to return credible 
    intervals for differnetial rate

    Parameters
    ----------
    pdfs: Posterior density
    rate: Merger rate
    intervals: percentiles in intervals
    """
    
    dRd = rates * np.array(pdfs).T
    if np.isscalar(intervals):
        return np.percentile(dRd, intervals, axis = 1)
    cred_vals = []
    for cred in intervals:
        cred_vals.append(np.percentile(dRd, cred, axis = 1))
    
    return cred_vals