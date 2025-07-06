import numpy as np
import models, gnobs
from conversions import *
            
def get_data_for_fitting(pe, inj):
    """ 
    Organise the data for faster analysis

    Parameters
    ----------
    pe : Dictionary containing parameter estimates of all the observation runs
    inj : Dictionary containing injections performed over all the observation runs

    Returns
    -------
    Organised pe and injection data.
    """
    
    data = {}
    injections = inj.copy()
    for obsrun in pe.keys():
        redshifts = []
        breakat, breaks = 0, [0]
        mass1, mass2 = [], []
        spin1z, spin2z = [], []
        pe_prior_pdf = []
        
        for key in pe[obsrun].keys():
            m1, m2 = pe[obsrun][key]['mass1_src'], pe[obsrun][key]['mass2_src']
            s1z, s2z = pe[obsrun][key]['spin1z'], pe[obsrun][key]['spin2z']
            mass1 = np.append(mass1, m1)
            mass2 = np.append(mass2, m2)
            spin1z = np.append(spin1z, s1z)
            spin2z = np.append(spin2z, s2z)
            breakat += len(m1)
            breaks.append(breakat)
            redshifts = np.append(redshifts, pe[obsrun][key]['redshift'])
            pe_prior_pdf.extend(pe[obsrun][key]['prior_pdf'])
        
        obsrun_data = {}
        obsrun_data['parametric_data'] = np.transpose([mass1, mass2, spin1z, spin2z])
        obsrun_data['breaks'] = breaks
        #obsrun_data['redshifts'] = np.array(redshifts)
        obsrun_data['log1pz'] = np.log1p(redshifts)
        # Following is independent of analysis thus combining for faster computation
        obsrun_data['analysis_independent'] = z_to_dcovdz(redshifts) / (1 + redshifts)
        obsrun_data['analysis_independent'] /= np.array(pe_prior_pdf)
        
        data[obsrun] = obsrun_data
        
    for obsrun in inj.keys():
        m1_rec = injections[obsrun]['mass1_rec']
        m2_rec = injections[obsrun]['mass2_rec']
        s1z_rec = injections[obsrun]['s1z_rec']
        s2z_rec = injections[obsrun]['s2z_rec']
        z_rec = injections[obsrun]['z_rec']
        w_rec = injections[obsrun]['w_rec']
        var_rec = np.transpose([m1_rec, m2_rec, s1z_rec, s2z_rec])
        injections[obsrun]['var_rec'] = var_rec
        injections[obsrun]['w_rec'] = w_rec
        injections[obsrun]['log1pz'] = np.log1p(z_rec)
        injections[obsrun]['analysis_independent'] = z_to_dcovdz(z_rec) / (1 + z_rec)
        injections[obsrun]['analysis_independent'] /= injections[obsrun]['rec_pdf']
        
    return data, injections

def get_binned_data_for_fitting(pe, inj, zsegs):
    """ 
    Organise the data for faster binned analysis

    Parameters
    ----------
    pe : Dictionary containing parameter estimates of all the observation runs
    inj : Dictionary containing injections performed over all the observation runs
    zsegs: Redshift segements

    Returns
    -------
    Organised pe and injection data.
    """
    
    data = {}
    injections = inj.copy()
    for obsrun in pe.keys():
        obsrun_data = {}
        
        for obs in pe[obsrun].keys():
            obsrun_data[obs] = {}
            parametric_data = []
            qarr, logq, pe_prior_pdf = [], [], []
            log1pz, analysis_independent = [], []
            
            mch, q = pe[obsrun][obs]['mchirp_src'], pe[obsrun][obs]['q']
            s1z, s2z = pe[obsrun][obs]['spin1z'], pe[obsrun][obs]['spin2z']
            z = pe[obsrun][obs]['redshift']
            prior_pdf = pe[obsrun][obs]['prior_pdf']
            
            for ii, _ in enumerate(zsegs[:-1]):
                idxsel = np.where((z >= zsegs[ii]) & (z < zsegs[ii + 1]))
                parametric_data.append(np.transpose([mch[idxsel], s1z[idxsel], s2z[idxsel]]))
                log1pz.append(np.log1p(z[idxsel]))
                qarr.append(q[idxsel])
                logq.append(np.log(q[idxsel]))
                pe_prior_pdf.append(prior_pdf[idxsel])
                ai = z_to_dcovdz(z[idxsel]) / (1 + z[idxsel]) / prior_pdf[idxsel]
                analysis_independent.append(ai)
        
            obsrun_data[obs]['parametric_data'] = parametric_data
            obsrun_data[obs]['logq'] = logq
            obsrun_data[obs]['q'] = qarr
            obsrun_data[obs]['log1pz'] = log1pz
            obsrun_data[obs]['analysis_independent'] = analysis_independent
        
        data[obsrun] = obsrun_data
    
    inj_data = {}
    for obsrun in inj.keys():
        mch = injections[obsrun]['mch_rec']
        s1z = injections[obsrun]['s1z_rec']
        s2z = injections[obsrun]['s2z_rec']
        z = injections[obsrun]['z_rec']
        q = injections[obsrun]['q_rec']
        prior_pdf = injections[obsrun]['rec_pdf']
        
        inj_data[obsrun] = {}
        parametric_data = []
        qarr, logq = [], []
        log1pz, analysis_independent = [], []
        
        for ii, _ in enumerate(zsegs[:-1]):
            idxsel = np.where((z >= zsegs[ii]) & (z < zsegs[ii + 1]))
            parametric_data.append(np.transpose([mch[idxsel], s1z[idxsel], s2z[idxsel]]))
            log1pz.append(np.log1p(z[idxsel]))
            qarr.append(q[idxsel])
            logq.append(np.log(q[idxsel]))
            ai = z_to_dcovdz(z[idxsel]) / (1 + z[idxsel]) / prior_pdf[idxsel]
            analysis_independent.append(ai)
                
        inj_data[obsrun]['parametric_data'] = parametric_data
        inj_data[obsrun]['logq'] = logq
        inj_data[obsrun]['q'] = qarr
        inj_data[obsrun]['log1pz'] = log1pz
        inj_data[obsrun]['analysis_independent'] = analysis_independent
        
        inj_data[obsrun]['analysis_time_yr'] = injections[obsrun]['analysis_time_yr']
        inj_data[obsrun]['ndraw'] = injections[obsrun]['ndraw']
        
    return data, inj_data