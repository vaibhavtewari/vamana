import numpy as np
import models, gnobs
from conversions import *

def reweight_data_to_UinComov(pe, inj):
    """ 
    Re-weights the data to uniform-in-comoving.

    Parameters
    ----------
    pe : Dictionary containing parameter estimates of all the observation runs
    inj : Dictionary containing injections performed over all the observation runs

    Returns
    -------
    Copy of dictionaries re-weighted to uniform in comoving distribution 
        for the redshifts.
    """
    sfr_model = models.UinComov
    
    for obsrun in pe.keys():
        for event in pe[obsrun].keys():
            z = pe[obsrun][event]['redshift']
            z_max = inj[obsrun]['z_max']
            dNdz = gnobs.get_sfr_pdf(sfr_model, z, z_max = z_max, normalise = False)
            pe[obsrun][event]['prior_pdf'] /= dNdz

    for obsrun in inj.keys():
        z_max = inj[obsrun]['z_max']
        z_rec = inj[obsrun]['z_rec']
        pdf_sfr = gnobs.get_sfr_pdf(sfr_model, z_rec, z_max = z_max, normalise = True)
        inj[obsrun]['rec_pdf'] /= pdf_sfr
    
        inj[obsrun]['surveyed_VT'] = gnobs.get_dNdz_norm(sfr_model, z_max = z_max)
        inj[obsrun]['surveyed_VT'] *= inj[obsrun]['analysis_time_yr']
        inj[obsrun]['surveyed_VT'] /= 1e9
            
    return pe, inj
            
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
        mchirp, spin1z, spin2z = [], [], []
        qarr, pe_prior_pdf = [], []
        
        for key in pe[obsrun].keys():
            mch, q = pe[obsrun][key]['mchirp_src'], pe[obsrun][key]['q']
            s1z, s2z = pe[obsrun][key]['spin1z'], pe[obsrun][key]['spin2z']
            mchirp = np.append(mchirp, mch)
            spin1z = np.append(spin1z, s1z)
            spin2z = np.append(spin2z, s2z)
            breakat += len(mch)
            breaks.append(breakat)
            redshifts = np.append(redshifts, pe[obsrun][key]['redshift'])
        
            qarr.extend(pe[obsrun][key]['q'])
            pe_prior_pdf.extend(pe[obsrun][key]['prior_pdf'])
        
        obsrun_data = {}
        obsrun_data['parametric_data'] = np.transpose([mchirp, spin1z, spin2z])
        obsrun_data['logq'] = np.log(qarr)
        obsrun_data['q'] = np.array(qarr)
        obsrun_data['breaks'] = breaks
        #obsrun_data['redshifts'] = np.array(redshifts)
        obsrun_data['log1pz'] = np.log1p(redshifts)
        # Following is independent of analysis thus combining for faster computation
        obsrun_data['analysis_independent'] = z_to_dcovdz(redshifts) / (1 + redshifts)
        obsrun_data['analysis_independent'] /= np.array(pe_prior_pdf)
        
        data[obsrun] = obsrun_data
        
    for obsrun in inj.keys():
        mch_rec = injections[obsrun]['mch_rec']
        s1z_rec = injections[obsrun]['s1z_rec']
        s2z_rec = injections[obsrun]['s2z_rec']
        z_rec = injections[obsrun]['z_rec']
        var_rec = np.transpose(np.array([mch_rec, s1z_rec, s2z_rec]))
        injections[obsrun]['var_rec'] = var_rec
        injections[obsrun]['log1pz'] = np.log1p(z_rec)
        injections[obsrun]['analysis_independent'] = z_to_dcovdz(z_rec) / (1 + z_rec)
        injections[obsrun]['analysis_independent'] /= injections[obsrun]['rec_pdf']
        
    return data, injections