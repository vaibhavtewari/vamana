import os
import glob
import numpy as np
import h5py

from vamana.conversions import *


def _get_pe_weights(mass1, mass2, spin1z, spin2z, lumd):
    """
    Compute PE prior weights for GWTC-1 events.

    LALInference uses a uniform prior in detector-frame masses and a
    uniform or log-uniform prior in aligned spin components. This function
    computes the corresponding prior PDF in source-frame parameters.

    Parameters
    ----------
    mass1 : np.ndarray
        Detector-frame primary mass.
    mass2 : np.ndarray
        Detector-frame secondary mass.
    spin1z : np.ndarray
        Primary aligned spin component.
    spin2z : np.ndarray
        Secondary aligned spin component.
    lumd : np.ndarray
        Luminosity distance [Mpc].

    Returns
    -------
    np.ndarray
        Prior probability density evaluated at each sample.
    """
    z = dlum_to_z(lumd)
    ps1z = -np.log(np.abs(spin1z)) / 2
    ps2z = -np.log(np.abs(spin2z)) / 2
    prior_pdf = ps1z * ps2z
    psrc = lumd ** 2
    psrc *= (1 + z) ** 2
    psrc *= get_dLdz(z)
    prior_pdf *= psrc
    return prior_pdf

def _get_pe_weights_uincomov(mass1_src, mass2_src,  spin1z, spin2z, z):
    
    ''' 
        Calculate PE prior density
        mass1_src: Source frame primary mass
        mass1_src: Source frame secondary mass
        spin1z: Aligned spin component for primary mass
        spin2z: Aligned spin component for secondary mass
        z: Redshift
    '''
    
    #Spins are uniform in magnitude and isotropic orientation
    #Following is density for the aligned spin components
    # Keep the in-plane spin unchanged
    ps1z = - np.log(np.abs(spin1z)) / 2
    ps2z = - np.log(np.abs(spin2z)) / 2
    prior_pdf = ps1z * ps2z
    
    # PE uses uniform in comoving, suppressed by a dilation factor of (1+z)
    # see https://lscsoft.docs.ligo.org/bilby/api/bilby.gw.prior.UniformSourceFrame.html
    psrc = z_to_dcovdz(z)
    psrc /= (1 + z)
    prior_pdf *= psrc
    
    #change from m1-m2 to mch_src-q in source frame
    J = (1 + z) ** 2 #detector m1-m2 to source m1-m2
    prior_pdf *= J
     
    return prior_pdf

def read_pe_gwtc1(pe_dir, nsamp):
    """
    Read GWTC-1 posterior samples from files matching a glob pattern.

    Reads mass, spin, and luminosity distance parameters from each file,
    computes derived quantities (chirp mass, mass ratio, redshift), and
    draws a random subsample of size nsamp from each event.

    Parameters
    ----------
    pe_dir : str
        Glob pattern pointing to the PE files e.g. "/data/GWTC-1/*bbh*".
    nsamp : int
        Number of posterior samples to draw from each event.

    Returns
    -------
    dict
        Nested dictionary keyed by event name. Each value is a dictionary
        containing:
        - 'mass1' : np.ndarray, detector-frame primary mass
        - 'mass2' : np.ndarray, detector-frame secondary mass
        - 'mass1_src' : np.ndarray, source-frame primary mass
        - 'mass2_src' : np.ndarray, source-frame secondary mass
        - 'spin1z' : np.ndarray, primary aligned spin component
        - 'spin2z' : np.ndarray, secondary aligned spin component
        - 'lumd' : np.ndarray, luminosity distance [Mpc]
        - 'mchirp_src' : np.ndarray, source-frame chirp mass
        - 'q' : np.ndarray, mass ratio mass2/mass1
        - 'redshift' : np.ndarray, redshift derived from luminosity distance
        - 'prior_pdf' : np.ndarray, PE prior probability density

    Raises
    ------
    FileNotFoundError
        If no files match the glob pattern.

    Examples
    --------
    >>> pe = read_gwtc1("/data/GWTC-1/*bbh*", nsamp=5000)
    >>> pe['GW150914']['mass1_src']
    array([...])
    """
    pe_files = np.sort(glob.glob(pe_dir))
    if len(pe_files) == 0:
        raise FileNotFoundError(f"No files found matching: {pe_dir}")

    pe = {}
    for ff in pe_files:
        with h5py.File(ff, 'r') as bulk:
            super_event = ff.rsplit("/")[-1][:-5]
            pe[super_event] = {}

            for key in ['mass_1', 'm1']:
                try:
                    pe[super_event]['mass1'] = bulk[key][:]
                except:
                    pass
            for key in ['mass_2', 'm2']:
                try:
                    pe[super_event]['mass2'] = bulk[key][:]
                except:
                    pass
            for key in ['mass_1_source', 'm1_source']:
                try:
                    pe[super_event]['mass1_src'] = bulk[key][:]
                except:
                    pass
            for key in ['mass_2_source', 'm2_source']:
                try:
                    pe[super_event]['mass2_src'] = bulk[key][:]
                except:
                    pass
            for key in ['spin_1z', 'a1z']:
                try:
                    pe[super_event]['spin1z'] = bulk[key][:]
                except:
                    pass
            for key in ['spin_2z', 'a2z']:
                try:
                    pe[super_event]['spin2z'] = bulk[key][:]
                except:
                    pass
            for key in ['luminosity_distance', 'dist']:
                try:
                    pe[super_event]['lumd'] = bulk[key][:]
                except:
                    pass

            pe[super_event]['mchirp_src'] = (
                (pe[super_event]['mass1_src'] * pe[super_event]['mass2_src']) ** 0.6
                / (pe[super_event]['mass1_src'] + pe[super_event]['mass2_src']) ** 0.2
            )
            pe[super_event]['q'] = (
                pe[super_event]['mass2_src'] / pe[super_event]['mass1_src']
            )
            prior_pdf = _get_pe_weights(
                pe[super_event]['mass1_src'],
                pe[super_event]['mass2_src'],
                pe[super_event]['spin1z'],
                pe[super_event]['spin2z'],
                pe[super_event]['lumd']
            )
            pe[super_event]['prior_pdf'] = prior_pdf
            pe[super_event]['redshift'] = dlum_to_z(pe[super_event]['lumd'])

            npe = len(prior_pdf)
            idxsel = np.random.choice(np.arange(npe), size=nsamp, replace=False)
            for key in pe[super_event].keys():
                pe[super_event][key] = pe[super_event][key][idxsel]

    return pe

def read_pe_o3(pe_dir, nsamp):

    pe_files = np.sort(glob.glob(pe_dir))
    pe, i = {}, 0
    for i, ff in enumerate(pe_files):
    
        with h5py.File(ff, 'r') as bulk:
            
            super_event = ff.rsplit("/")[-1][:-5]
            pe[super_event] = {}
            
            parameters = bulk.keys()
            
            for key in ['mass_1', 'm1']:
                try:
                    pe[super_event]['mass1'] = bulk[key][:]
                except:
                    pass
            for key in ['mass_2', 'm2']:
                try:
                    pe[super_event]['mass2'] = bulk[key][:]
                except:
                    pass
            for key in ['mass_1_source', 'm1_source']:
                try:
                    pe[super_event]['mass1_src'] = bulk[key][:]
                except:
                    pass
            for key in ['mass_2_source', 'm2_source']:
                try:
                    pe[super_event]['mass2_src'] = bulk[key][:]
                except:
                    pass
            for key in ['spin_1z', 'a1z']:
                try:
                    pe[super_event]['spin1z'] = bulk[key][:]
                except:
                    pass
            for key in ['spin_2z', 'a2z']:
                try:
                    pe[super_event]['spin2z'] = bulk[key][:]
                except:
                    pass
            for key in ['luminosity_distance', 'dist']:
                try:
                    pe[super_event]['lumd'] = bulk[key][:]
                except:
                    pass
            pe[super_event]['mchirp_src'] = (pe[super_event]['mass1_src'] * pe[super_event]['mass2_src']) ** 0.6 
            pe[super_event]['mchirp_src'] /= (pe[super_event]['mass1_src'] + pe[super_event]['mass2_src']) ** 0.2
            pe[super_event]['q'] = pe[super_event]['mass2_src'] / pe[super_event]['mass1_src']
            prior_pdf = _get_pe_weights(pe[super_event]['mass1_src'], pe[super_event]['mass2_src'],\
                 pe[super_event]['spin1z'], pe[super_event]['spin2z'], pe[super_event]['lumd'])
            pe[super_event]['prior_pdf'] = prior_pdf
            pe[super_event]['redshift'] = dlum_to_z(pe[super_event]['lumd'])
            
            npe = len(prior_pdf)
            idxsel = np.random.choice(np.arange(npe), size = nsamp, replace=False)
            for key in pe[super_event].keys():
                pe[super_event][key] = pe[super_event][key][idxsel]
    
    return pe


def read_pe_o4(pe_dir, nsamp):

    pe = {}
    pe_files = np.sort(glob.glob(pe_dir))
    for pefile in pe_files:
        #print (pefile)
        fname = pefile.split('/')[-1][:-3]
        pe[fname] = {}
        ff = h5py.File(pefile, 'r')
    
        key = list(ff.keys())
        #print (key)
        boolean = (np.char.find(key, 'online') >= 0)
        boolean += (np.char.find(key, 'result') >= 0)
        boolean += (np.char.find(key, 'merge') >= 0)
        boolean += (np.char.find(key, 'Exp0') >= 0)
        boolean += (np.char.find(key, 'EXP6') >= 0)
        boolean += (np.char.find(key, 'Exp6') >= 0)
        boolean += (np.char.find(key, 'Exp3') >= 0)
        boolean += (np.char.find(key, 'EXP0') >= 0)
        boolean += (np.char.find(key, 'PROD8') >= 0)
        boolean += (np.char.find(key, 'rift-v5PHM-calmarg') >= 0)
        boolean += (np.char.find(key, 'bilby-IMRPhenomXPHM-SpinTaylor') >= 0)
        boolean += (np.char.find(key, 'bilby-IMRPhenomXPHM-SpinTaylor-2') >= 0)
        boolean += (np.char.find(key, 'bilby-IMRPhenomXPHM-SpinTaylor-3') >= 0)
        boolean += (np.char.find(key, 'bilby-IMRPhenomXPHM-SpinTaylor-4') >= 0)
        boolean += (np.char.find(key, 'bilby-IMRPhenomXPHM-SpinTaylor-5') >= 0)
        boolean += (np.char.find(key, 'bilby-IMRPhenomXPHM-SpinTaylor-6') >= 0)
        boolean += (np.char.find(key, 'bilby-SEOBNRv5PHM') >= 0)
        boolean += (np.char.find(key, 'bilby-SEOBNRv5PHM-2') >= 0)
        boolean += (np.char.find(key, 'bilby-SEOBNRv5PHM-3') >= 0)
        boolean += (np.char.find(key, 'bilby-SEOBNRv5PHM-4') >= 0)
        boolean += (np.char.find(key, 'bilby-SEOBNRv5PHM-5') >= 0)
        boolean += (np.char.find(key, 'bilby-SEOBNRv5PHM-6') >= 0)
        boolean += (np.char.find(key, 'rift-v5PHM-calmarg') >= 0)
        boolean += (np.char.find(key, 'bilby-IMRPhenomXPNR') >= 0)
        boolean += (np.char.find(key, 'posterior') >= 0)
        boolean += (np.char.find(key, 'bilby-IMRPhenomXO4a-2') >= 0)
        boolean += (np.char.find(key, 'bilby-NRSur7dq4') >= 0)
        boolean += (np.char.find(key, 'Online') >= 0)
        boolean += (np.char.find(key, 'Exp1') >= 0)
        boolean += (np.char.find(key, 'EXP1') >= 0)
        boolean += (np.char.find(key, 'Exp2') >= 0)
        boolean += (np.char.find(key, 'posterior') >= 0)
        boolean += (np.char.find(key, 'C00:Mixed') >= 0)
        boolean += (np.char.find(key, 'G489286') >= 0)
        key = key[np.where(boolean == True)[0][0]]
        
        if key == 'posterior':
            res = ff['posterior']
        else:
            res = ff[key]['posterior_samples'][()]
        pe[fname]['mass1_src'] = res['mass_1_source'][()]
        pe[fname]['mass2_src'] = res['mass_2_source'][()]
        pe[fname]['mass1'] = res['mass_1'][()]
        pe[fname]['mass2'] = res['mass_2'][()]
        pe[fname]['spin1z'] = res['spin_1z'][()]
        pe[fname]['spin2z'] = res['spin_2z'][()]
        pe[fname]['redshift'] = res['redshift'][()]
    
        pe[fname]['q'] = res['mass_ratio'][()]
        pe[fname]['mchirp_src'] = res['chirp_mass_source'][()]
    
        prior_pdf = _get_pe_weights_uincomov(pe[fname]['mass1_src'], pe[fname]['mass2_src'],\
                 pe[fname]['spin1z'], pe[fname]['spin2z'], pe[fname]['redshift'])
        pe[fname]['prior_pdf'] = prior_pdf
        pe[fname]['redshift'] = pe[fname]['redshift']

        npe = len(prior_pdf)
        idxsel = np.random.choice(np.arange(npe), size = nsamp, replace=False)
        for key in pe[fname].keys():
            pe[fname][key] = pe[fname][key][idxsel]
    
        ff.close()
    return pe

def read_injections_gwtc1(fin, DETSNR_THR, NETSNR_THR):
    """
    Read an injection file from a GWTC-1 observation run.

    Applies SNR thresholds to select detected injections, computes
    redshifts from chirp mass ratios, and assembles the injection
    prior PDF from individual parameter draws.

    Parameters
    ----------
    fin : str
        Full path to the injections HDF5 file.
    DETSNR_THR : float
        Minimum per-detector SNR threshold for an injection to be
        considered detected (applied to H1 and L1).
    NETSNR_THR : float
        Minimum network SNR threshold for an injection to be
        considered detected.

    Returns
    -------
    dict
        Dictionary containing:
        - 'analysis_time_yr' : float, total analysis time in years
        - 'ndraw' : float, total number of injections drawn
        - 'mass1_rec' : np.ndarray, recovered source-frame primary mass
        - 'mass2_rec' : np.ndarray, recovered source-frame secondary mass
        - 's1z_rec' : np.ndarray, recovered primary aligned spin
        - 's2z_rec' : np.ndarray, recovered secondary aligned spin
        - 'z_rec' : np.ndarray, recovered redshift
        - 'rec_pdf' : np.ndarray, injection prior PDF
        - 'w_rec' : np.ndarray, injection weights (ones for GWTC-1)

    Raises
    ------
    FileNotFoundError
        If the injections file does not exist.

    Examples
    --------
    >>> inj = read_injections_gwtc1("/data/injections/o1o2.h5", DETSNR_THR=8.0, NETSNR_THR=12.0)
    >>> inj['mass1_rec']
    array([...])
    """
    injections = {}
    with h5py.File(fin, 'r') as inp:
        secs_in_year = 365.25 * 86400
        injections['analysis_time_yr'] = inp.attrs['total_analysis_time'] / secs_in_year
        injections['ndraw'] = inp.attrs['total_generated']
        max_spin = 0.99

        min_DETSNR = 1e5
        for key in ['snr_H', 'snr_L']:
            min_DETSNR = np.minimum(inp['events'][key], min_DETSNR)
        NETSNR = inp['events']['snr_net']
        idxsel = np.where((min_DETSNR > DETSNR_THR) & (NETSNR > NETSNR_THR))

        m1_rec  = inp['events']['mass1_source'][idxsel]
        m2_rec  = inp['events']['mass2_source'][idxsel]
        s1z_rec = inp['events']['spin1z'][idxsel]
        s2z_rec = inp['events']['spin2z'][idxsel]
        z_rec   = inp['events']['Mc'][idxsel] / inp['events']['Mc_source'][idxsel] - 1

        pz  = inp['events']['logpdraw_z'][idxsel]
        pm1 = inp['events']['logpdraw_mass1_source_GIVEN_z'][idxsel]
        pm2 = inp['events']['logpdraw_mass2_source_GIVEN_mass1_source'][idxsel]
        ps1z = (np.log(max_spin) - np.log(np.abs(s1z_rec))) / 2 / max_spin
        ps2z = (np.log(max_spin) - np.log(np.abs(s2z_rec))) / 2 / max_spin
        rec_pdf = np.exp(pm1 + pm2 + pz) * ps1z * ps2z

        injections['mass1_rec'] = m1_rec
        injections['mass2_rec'] = m2_rec
        injections['s1z_rec']   = s1z_rec
        injections['s2z_rec']   = s2z_rec
        injections['z_rec']     = z_rec
        injections['rec_pdf']   = rec_pdf
        injections['w_rec']     = np.ones_like(rec_pdf)

    return injections

def read_injections_o3(fin, IFAR_thr):
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
        idxsel = np.where(max_IFAR > IFAR_thr)
            
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


def read_injections_o4(fin, IFAR_thr, gps_start, gps_stop):
    ''' 
        Read injection data
        fin: location of h5py file
        idx: index of injections that crossed a selection criteria
        
    '''

    ff = h5py.File(fin, 'r')
    data = ff['events']
    dtypes = np.array([d[0] for d in data.dtype.descr])
    gps = data['time_geocenter']
    max_ifar = 0
    for dtype in dtypes:
        if 'far' in dtype:
            ifar = 1 / data[dtype]
            max_ifar = np.maximum(max_ifar, ifar)

    idxsel = np.where((max_ifar >= IFAR_thr) & (gps >= gps_start) & (gps < gps_stop))
    
    max_spin = 1.0
    injections = {}
        
    secs_in_year = 365.25 * 86400
    injections['analysis_time_yr'] = ff.attrs['total_analysis_time']
    injections['analysis_time_yr'] /= secs_in_year
    injections['ndraw'] = ff.attrs['total_generated']
            
    data = ff['events']
    m1_rec = data['mass1_source'][idxsel]
    m2_rec = data['mass2_source'][idxsel]
    s1x_rec = data['spin1x'][idxsel]
    s2x_rec = data['spin2x'][idxsel]
    s1y_rec = data['spin1y'][idxsel]
    s2y_rec = data['spin2y'][idxsel]
    s1z_rec = data['spin1z'][idxsel]
    s2z_rec = data['spin2z'][idxsel]
    z_rec = data['z'][idxsel]
    w_rec = data['weights'][idxsel]
        
    rec_log_pm1 = data['lnpdraw_mass1_source'][idxsel]
    rec_log_pm2 = data['lnpdraw_mass2_source_GIVEN_mass1_source'][idxsel]
    rec_log_pz = data['lnpdraw_z'][idxsel]
    
    ps1z = (np.log(max_spin) - np.log(np.abs(s1z_rec))) / 2 / max_spin
    ps2z = (np.log(max_spin) - np.log(np.abs(s2z_rec))) / 2 / max_spin
    rec_pdf = np.exp(rec_log_pm1 + rec_log_pm2 + rec_log_pz) * ps1z * ps2z
        
    injections['mass1_rec'] = m1_rec
    injections['mass2_rec'] = m2_rec
    injections['s1z_rec'] = s1z_rec
    injections['s2z_rec'] = s2z_rec
    injections['z_rec'] = z_rec
    injections['rec_pdf'] = rec_pdf
    injections['w_rec'] = w_rec
        
    return injections