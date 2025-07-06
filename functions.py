import os, sys, copy, h5py

import numpy as np
import scipy.stats as ss
from scipy.special import logsumexp
from scipy.interpolate import interp1d

from multiprocessing import Pool, cpu_count
from scipy.stats import norm, chi2

import gnobs
from models import *
from conversions import *

def get_zsegs(pe, nsegs):
    """
    Estimate approximate segments in redshift that have contributed equally
        to the number of observations
    pe: Parameter estimates of all the runs
    nsegs: Number of segments
    Returns
    -------
    zsegs: The redshift segments
    zseg_filled: list detailing which of these segments are filled by each
        observation
    """
    
    zax = np.linspace(0, 2.0, 1000)
    c_zax = 0.5 * (zax[:-1] + zax[1:])
    hist = 0
    for obsrun in pe.keys():
        for obs in pe[obsrun].keys():
            prior_pdf = pe[obsrun][obs]['prior_pdf']
            z = pe[obsrun][obs]['redshift']
            mch = pe[obsrun][obs]['mchirp_src']
            #Re-weight to an approximate population -- not much bearing
            pout = gnobs.get_sfr_pdf(sfr_1pz, z, \
                                     kappa = 2.5, z_max = np.max(z))
            pout *= powerlaw_pdf(mch, 5., 60., 3.0)
            h, _ = np.histogram(z, weights = pout/prior_pdf, \
                                density = True, bins = zax)
            hist += h
    hist = np.cumsum(hist)
    hist /= hist[-1]
    zsegs = [0]
    for ii in range(1, nsegs):
        zax_segmax = c_zax[hist <= ii/nsegs][-1]
        zsegs.append(zax_segmax)
    zsegs.append(zax[-1])
    
    zseg_filled = {}
    for obsrun in pe.keys():
        zseg_filled[obsrun] = []
        for obs in list(pe[obsrun].keys()):
            zsegfill = np.array([]).astype(int)
            z = pe[obsrun][obs]['redshift']
            w = 1 / pe[obsrun][obs]['lumd']
            sumw = np.sum(w)
            for ii in range(len(zsegs) - 1):
                idx = np.where((z >= zsegs[ii]) & (z < zsegs[ii+1]))
                if len(w[idx]) > 0:
                    zsegfill = np.append(zsegfill, ii)
                
            zseg_filled[obsrun].append(zsegfill)
    
    return zsegs, zseg_filled

def get_zsegs_idx(pe, injections, zsegs):
    """
    Identify the indexes of the posterior sample or injection as belonging
        to a redshift segment
    pe: Parameter estimates of all the runs
    injections: Injections for all the runs
    zsegs: The redshift segments
    Returns
    -------
    zseg_peidx: Array that stores index for measured redshift for observations
    zseg_injidx: Array that stores index for injected redshift for 
        injections
    """
    zseg_peidx = {}
    for obsrun in pe.keys():
        zsegpeidx = []
        for obs in list(pe[obsrun].keys()):
            idxx = []
            for ii in range(len(zsegs) - 1):
                z = pe[obsrun][obs]['redshift']
                idx = np.where((z >= zsegs[ii]) & (z <= zsegs[ii + 1]))
                idxx.append(idx)
            zsegpeidx.append(idxx)
            
        zseg_peidx[obsrun] = zsegpeidx
            
    zseg_injidx = {}
    for obsrun in injections.keys():
        z = injections[obsrun]['z_rec']
        zseg_injidx[obsrun] = []
        for ii in range(len(zsegs) - 1):
            idx = np.where((z >= zsegs[ii]) & (z < zsegs[ii + 1]))
            zseg_injidx[obsrun].append(idx)
            
    return zseg_peidx, zseg_injidx

def get_m1m2_spinz_params(locs_m1, stds_m1, locs_m2, stds_m2, corr_m1m2, locs_sz, stds_sz):
    """
    Organise 1D Gaussians in a mixture componet in a 3D Gaussian
    locs_m1: Location of Gaussians modeling primary mass
    stds_m1: Scale of Gaussians modeling primary mass
    locs_m2: Location of Gaussians modeling Secondary mass
    stds_m2: Scale of Gaussians modeling Secondary mass
    corr_m1m2: correlation factor between two mass parameter
    locs_sz: Location of Gaussians modeling aligned spins
    stds_sz: Scale of Gaussians modeling aligned spins
    Returns
    -------
    3D arrays for each component
    """
    
    ngauss, means, covs = len(locs_m1), [], []
    for ii in range(ngauss):
        mean = np.array([locs_m1[ii], locs_m2[ii], locs_sz[ii], locs_sz[ii]])
        cov = np.diag([stds_m1[ii] ** 2, stds_m2[ii] ** 2, stds_sz[ii] ** 2, stds_sz[ii] ** 2])
        
        cov[0][1] = cov[1][0] = stds_m1[ii] * stds_m2[ii] * corr_m1m2[ii]
        
        means.append(mean)
        covs.append(cov)
        
    return means, covs
    
def function_gauss(data_analysis):
    """
    Central function to sample and store
    data_analysis: Dictionary containing data for the analysis.
    
    Returns
    -------
    Saves files with posteriors
    """
    
    np.random.seed()
    
    pe = data_analysis['pe']
    injections = data_analysis['injections']
    analysis = data_analysis['analysis']
    args_sampler, args_ppd = analysis.define_args(pe, injections)
    ngauss = args_sampler['ngauss']
    
    sum_log_lkl = log_lkl = maxlkl = -1e8
    all_proposals, itr, stp = [], 0, 0
    hyperparams = analysis.initialise_hyperparams(args_sampler)
    filenum, nsampled_eff = 0, 0
    margl = []
    fname_prefix = sys.argv[-1] + '_' + analysis.__name__ + '_'

    while stp < args_ppd['niter']:

        itr += 1
        prp_hyperparams, mhr = \
                analysis.get_hyperparams_proposal(args_sampler, hyperparams)

        logsum_sumprob = analysis.loglikelihood(args_sampler, prp_hyperparams)
        ratio = np.exp(logsum_sumprob - log_lkl)
        ratio *= mhr
        logmhr = np.log(mhr) + logsum_sumprob
        sum_log_lkl = np.logaddexp(sum_log_lkl, logmhr)
        nsampled_eff += mhr
        
        rndn = np.random.random()
        if ratio > rndn:

            stp += 1
            log_lkl = logsum_sumprob
            maxlkl = max(maxlkl, log_lkl)
            avg_log_lkl = sum_log_lkl - np.log(nsampled_eff)
            margl.append(avg_log_lkl)
            sum_log_lkl, nsampled_eff = -1e8, 0.
            
            hyperparams = prp_hyperparams
            locs_m1, locs_m2 = hyperparams['locs_m1'], hyperparams['locs_m2']
            corr_m1m2 = hyperparams['corr_m1m2']
            #locs_sz, stds_sz = hyperparams['locs_sz'], hyperparams['stds_sz']
            gwts = hyperparams['gwts']
            rate = hyperparams['rate']
            hyperparams['log_lkl'] = logsum_sumprob
            print (itr, stp, np.round(log_lkl, 1), np.round(maxlkl, 2), np.round(rate, 2), np.round(hyperparams['kappa'], 2))
            print (np.round(hyperparams['norm_m1m2'], 2))
            print (np.round(locs_m1, 2), '--locs_m1')
            print (np.round(locs_m2, 2), '--locs_m2')
            #print (np.round(corr_m1m2, 2), '--corr_m1m2')
            #print (np.round(locs_sz, 2), '--sz loc')
            #print (np.round(stds_sz, 2), '--sz std')
            #print (np.round(gwts, 3))
            #print ()
            
            if stp > args_ppd['nstart'] and stp % args_ppd['nstride'] == 0:
                
                hyperparams['margl'] = logsumexp(margl[-args_ppd['nstride']:])
                hyperparams['margl'] -= np.log(args_ppd['nstride'])
                margl = []
                fname = args_ppd['output'] + '/' + fname_prefix
                fname += str(stp) + '_' + str(itr) + '.hdf5'
                ppd = analysis.postprocess(args_sampler, args_ppd, hyperparams)
                with h5py.File(fname, 'w') as out:
                    group = out.create_group('posteriors')
                    for key in hyperparams.keys():
                        if np.isscalar(hyperparams[key]):
                            d = hyperparams[key]
                            group.create_dataset(key, data = d, dtype='float32')
                        else:
                            try:
                                group.create_dataset(key, data = hyperparams[key])
                            except:
                                group.create_dataset(key, data = str(hyperparams[key]))
                    group = out.create_group('ppd')
                    for key in ppd.keys():
                        if np.isscalar(ppd[key]):
                            group.create_dataset(key, data = ppd[key], dtype='float32')
                        else:
                            group.create_dataset(key, data = ppd[key])

    return True

def sample_mixture(lnprob_gauss, ncpu, all_cpu_args):
    """
    Run multiple copies
    """
    
    ncpu = len(all_cpu_args)
    
    pool = Pool(processes = ncpu, maxtasksperchild = 100)
    results = pool.map(lnprob_gauss, all_cpu_args)
    pool.close()

    return results

def gauss_proposal(x, loc, scale, dcdf):
    """
    Make proposals for a uniform prior with support range sensitive to 
            the denity of a Gaussian (eq. 7 in arXiv:2006.15047)
    x: Current position
    loc: Location of the Gaussian
    scale: Scale of the Gaussian
    dcdf: Change in cumulative density over the support range
    
    Returns
    -------
    The next proposal, inverse of jump probability
    """
    
    rv = norm(loc = loc, scale = scale)
    cdf_now = rv.cdf(x)
    cdf_left  = max(cdf_now - dcdf, 0.01)
    cdf_right = min(cdf_now + dcdf, 0.99)
    x_left = rv.ppf(cdf_left)
    x_right = rv.ppf(cdf_right)
        
    return np.random.uniform(x_left, x_right), x_right - x_left

def get_vt_and_error_parametric_o3(injections, mchpl_range, alpha_mch, \
                                   qmin, alpha_q, locs_sz, scales_sz):
    """ 
    Calculate sensitive volume for the reference population
    injections: Dictionary containing injections performed over all 
        the observation runs
    Remaining variables: Described in section 2.3 of arXiv:2006.15047
    
    Returns
    -------
    The sensitive volume and the Poisson error
    """
    
    ngauss_sz = len(locs_sz)
    mch_rec, q_rec = injections['mch_rec'], injections['q_rec']
    s1z_rec, s2z_rec = injections['s1z_rec'], injections['s2z_rec']
    VT = injections['surveyed_VT']#includes T where triggers were not generated
    ndraw = injections['ndraw']
    rec_pdf = injections['rec_pdf']

    prob_mch = broken_powerlaw_pdf(mch_rec, mchpl_range, alpha_mch)
    prob_q = powerlaw_pdf(q_rec, qmin, 1.0, alpha_q)
    prob_s1z = prob_s2z = 0
    for jj in range(ngauss_sz):
        prob_s1z += norm.pdf(s1z_rec, loc = locs_sz[jj], scale = scales_sz[jj])
        prob_s2z += norm.pdf(s2z_rec, loc = locs_sz[jj], scale = scales_sz[jj])
    prob_s1z /= ngauss_sz
    prob_s2z /= ngauss_sz
    pout = prob_mch * prob_q * prob_s1z * prob_s2z
    w = pout/rec_pdf/ndraw
    sumw = np.sum(w)

    return VT * sumw, np.sqrt(np.sum(w ** 2)) / sumw

def get_vt_and_error_parametric_sfr_1pz(sfr_model, injections, mchpl_range, \
                        alpha_mch, qmin, alpha_q, locs_sz, scales_sz, kappa):
    
    z_max = injections['z_max']
    dNdz = gnobs.get_sfr_pdf(sfr_1pz, injections['z_rec'], z_max = z_max, \
                             kappa = kappa, normalise = False)
    dNdz *= injections['analysis_time_yr']
    dNdz /= 1e9
    
    ngauss_sz = len(locs_sz)
    mch_rec, q_rec = injections['mch_rec'], injections['q_rec']
    s1z_rec, s2z_rec = injections['s1z_rec'], injections['s2z_rec']
    ndraw = injections['ndraw']
    rec_pdf = injections['rec_pdf']
    
    prob_mch = broken_powerlaw_pdf(mch_rec, mchpl_range, alpha_mch)
    prob_q = powerlaw_pdf(q_rec, qmin, 1.0, alpha_q)
    prob_s1z = prob_s2z = 0
    for jj in range(ngauss_sz):
        prob_s1z += norm.pdf(s1z_rec, loc = locs_sz[jj], scale = scales_sz[jj])
        prob_s2z += norm.pdf(s2z_rec, loc = locs_sz[jj], scale = scales_sz[jj])
    prob_s1z /= ngauss_sz
    prob_s2z /= ngauss_sz
    pout = prob_mch * prob_q * prob_s1z * prob_s2z
    w = dNdz * pout / rec_pdf / ndraw
    vts = np.sum(w)
    
    return vts 

def read_results(fin):
    """ 
    Read a result file
    fin: The path to the file
    
    Returns
    -------
    Dictionary containing the data
    """
    with h5py.File(fin, 'r') as inp:
    
        run_data = {}
        key = 'args_sampler'
        run_data[key] = {}
        for k in inp[key].keys():
            run_data[key][k] = inp[key][k][()]
        key = 'args_ppd'
        run_data[key] = {}
        for k in inp[key].keys():
            run_data[key][k] = inp[key][k][()]
        key = 'posteriors'
        run_data[key] = {}
        for k in inp[key].keys():
            run_data[key][k] = inp[key][k][()]
        key = 'ppd'
        run_data[key] = {}
        for k in inp[key].keys():
            run_data[key][k] = inp[key][k][()]

    return run_data

def read_injections_o3(fin, IFAR_THR):
    """ 
    Read an injection file performed over an observation runs
    fin: The path to the file
    IFAR_THR: The threshold for injections flagged as observed
    
    Returns
    -------
    Dictionary containing the injections
    """
    
    injections = {}
    with h5py.File(fin, 'r') as inp:
        
        secs_in_year = 365.25 * 86400
        injections['z_max'] = inp['injections'].attrs['max_redshift']
        injections['analysis_time_yr'] = inp['injections'].attrs['analysis_time_s']
        injections['analysis_time_yr'] /= secs_in_year
        injections['ndraw'] = inp['injections'].attrs['n_rejected'] 
        injections['ndraw'] += inp['injections'].attrs['n_accepted']
        injections['surveyed_VT'] = inp['injections'].attrs['N_exp/R(z=0)']   
        max_spin1 = inp['injections'].attrs['max_spin1']
        
        max_IFAR = 0
        for key in inp['injections'].keys():
            if 'IFAR' in key or 'ifar' in key:
                max_IFAR = np.maximum(inp['injections'][key], max_IFAR)
        idxsel = np.where(max_IFAR > IFAR_THR)
            
        m1_rec = inp['injections']['mass1_source'][()][idxsel]
        m2_rec = inp['injections']['mass2_source'][()][idxsel]
        s1z_rec = inp['injections']['spin1z'][()][idxsel]
        s2z_rec = inp['injections']['spin2z'][()][idxsel]
        z_rec = inp['injections']['redshift'][()][idxsel]
        
        key = 'mass1_source_mass2_source_sampling_pdf'
        pm1m2 = inp['injections'][key][()][idxsel]
        pz = inp['injections']['redshift_sampling_pdf'][()][idxsel]
        #Aligned spin pdf in precessing injections
        ps1z = (np.log(max_spin1) - np.log(np.abs(s1z_rec))) / 2 / max_spin1
        ps2z = (np.log(max_spin1) - np.log(np.abs(s2z_rec))) / 2 / max_spin1
        rec_pdf = pm1m2 * pz * ps1z * ps2z
        
        injections['mass1_rec'] = m1_rec
        injections['mass2_rec'] = m2_rec
        injections['s1z_rec'] = s1z_rec
        injections['s2z_rec'] = s2z_rec
        injections['z_rec'] = z_rec
        injections['rec_pdf'] = rec_pdf
        injections['w_rec'] = np.ones_like(rec_pdf)
        
    return injections

def read_injections_o1o2(fin):
    """ 
    Read an injection file performed over an observation runs
    fin: The path to the file
    IFAR_THR: The threshold for injections flagged as observed
    spin_orientation: Spin model used by injections(aligned/precessing)
    
    Returns
    -------
    Dictionary containing the injections
    """
    
    injections = {}
    with h5py.File(fin, 'r') as inp:

        injections['z_max'] = inp.attrs['z_max']
        injections['analysis_time_yr'] = inp.attrs['analysis_time_yr']
        injections['ndraw'] = inp.attrs['ndraw']
        injections['surveyed_VT'] = inp.attrs['N_exp/R(z=0)']
            
        mch_rec = inp['injections']['mch_rec'][()]
        q_rec = inp['injections']['q_rec'][()]
        s1z_rec = inp['injections']['s1z_rec'][()]
        s2z_rec = inp['injections']['s2z_rec'][()]
        z_rec = inp['injections']['z_rec'][()]
        rec_pdf = inp['injections']['rec_pdf'][()]
        rec_pdf *= J_mchq_to_m1m2(mch_rec, q_rec)

        m1_rec, m2_rec = qmch_to_m1m2(mch_rec, q_rec)
        injections['mass1_rec'] = m1_rec
        injections['mass2_rec'] = m2_rec
        injections['s1z_rec'] = s1z_rec
        injections['s2z_rec'] = s2z_rec
        injections['z_rec'] = z_rec
        injections['rec_pdf'] = rec_pdf
        injections['w_rec'] = np.ones_like(rec_pdf)
        
    return injections

def read_injections_o1o2_rnp(fin, DETSNR_THR, NETSNR_THR):
    """ 
    Read an injection file performed over an observation runs
    fin: The path to the file
    DETSNR_THR: The detector SNR threshold for injections flagged as observed
    NETSNR_THR: The network SNR threshold for injections flagged as observed
    
    Returns
    -------
    Dictionary containing the injections
    """
    
    injections = {}
    with h5py.File(fin, 'r') as inp:
        
        secs_in_year = 365.25 * 86400
        injections['analysis_time_yr'] = inp.attrs['total_analysis_time']
        injections['analysis_time_yr'] /= secs_in_year
        injections['ndraw'] = inp.attrs['total_generated']
        max_spin = 0.99
        
        min_DETSNR = 1e5
        for key in ['snr_H', 'snr_L']:
            min_DETSNR = np.minimum(inp['events'][key], min_DETSNR)
        NETSNR = inp['events']['snr_net']
        idxsel = np.where((min_DETSNR > DETSNR_THR) & (NETSNR > NETSNR_THR))
            
        m1_rec = inp['events']['mass1_source'][idxsel]
        m2_rec = inp['events']['mass2_source'][idxsel]
        s1z_rec = inp['events']['spin1z'][idxsel]
        s2z_rec = inp['events']['spin2z'][idxsel]
        z_rec = inp['events']['Mc'][idxsel] / inp['events']['Mc_source'][idxsel] - 1
        
        pz = inp['events']['logpdraw_z'][idxsel]
        key = 'logpdraw_mass1_source_GIVEN_z'
        pm1 = inp['events'][key][idxsel]
        key = 'logpdraw_mass2_source_GIVEN_mass1_source'
        pm2 = inp['events'][key][idxsel]
        #Aligned spin pdf in precessing injections
        ps1z = (np.log(max_spin) - np.log(np.abs(s1z_rec))) / 2 / max_spin
        ps2z = (np.log(max_spin) - np.log(np.abs(s2z_rec))) / 2 / max_spin
        rec_pdf = np.exp(pm1 + pm2 + pz) * ps1z * ps2z
        
        injections['mass1_rec'] = m1_rec
        injections['mass2_rec'] = m2_rec
        injections['s1z_rec'] = s1z_rec
        injections['s2z_rec'] = s2z_rec
        injections['z_rec'] = z_rec
        injections['rec_pdf'] = rec_pdf
        injections['w_rec'] = np.ones_like(rec_pdf)
        
    return injections

def get_dof(min_nu, max_nu, nsamp):
    """ 
    Generate dofs such that fractional change per dof density is roughly 
        constant -- generate fewer dofs that cause greater fractional 
        change and more dofs that cause smaller fractional change
    min_nu: Minimum value of the dof
    max_nu: Maximum value of the dof
    nsamp: Number of dofs
    
    Returns
    -------
    dof values
    """

    ax = np.linspace(min_nu, max_nu, nsamp).astype(int)
    chi_std = []
    for a in ax:
        std = chi2.ppf(0.5 + 0.3413447, df = a, scale = 1./a) 
        std -= chi2.ppf(0.5 - 0.3413447, df = a, scale = 1./a)
        std /= 2
        chi_std = np.append(chi_std, std)
        
    cdf = np.cumsum(chi_std)
    cdf /= cdf[-1]
    
    interp = interp1d(cdf, ax, bounds_error = False, fill_value = 'extrapolate')
    in_chidf = interp(np.linspace(min(cdf), 1, nsamp)).astype(int)
    
    return in_chidf

def get_mchirp_logprior(min_mch, max_mch, mch_ax, ngauss, nsamp = 40000):
    """
    Get the prior chirp mass distribution for an analysis
    
    Parameters
    ----------
    min_mch: Minimum chirp mass
    max_mch: Maximum chirp mass
    mch_ax: Axis on which to calculate prior's density
    ngauss: Number of components
    nsamp: Number of prior density samples
    
    Returns
    -------
    logprior:Logpdf of prior, nsamp in number
    """
    logprior = []
    fac = 0.15 * np.sqrt(15/ngauss)#hard coded as defined in proposals.py
    log_minmch, log_maxmch = np.log(min_mch), np.log(max_mch)
    for ii in range(nsamp):
        prp_loc = np.exp(np.random.uniform(log_minmch, log_maxmch, ngauss))
        minstd, maxstd = 0.02 * prp_loc, fac * prp_loc
        prp_std = np.random.uniform(minstd, maxstd, ngauss)
        gwts = ss.dirichlet.rvs(alpha = np.ones(ngauss))[0]
        logpdf = -np.inf
        for jj in range(ngauss):
            logpdf = np.logaddexp(logpdf, ss.norm.logpdf(mch_ax, loc = prp_loc[jj], scale = prp_std[jj]) + np.log(gwts[jj]))
        logprior.append(logpdf)
    return np.array(logprior)