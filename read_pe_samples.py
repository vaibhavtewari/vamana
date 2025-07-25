import glob, h5py
import numpy as np

from conversions import *

def read_pesamples(pe_dir, ifar_thr, nsel):

    pe_files = np.sort(glob.glob(pe_dir))
    pe, i = {}, 0
    for i, ff in enumerate(pe_files):
    
        with h5py.File(ff, 'r') as bulk:
            
            super_event = ff.rsplit("/")[-1][:-5]
            ifar = [int(s) for s in super_event.split('_') if s.isdigit()][-1]
            if ifar < ifar_thr:
                continue
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
            prior_pdf = get_pe_weights(pe[super_event]['mass1_src'], pe[super_event]['mass2_src'],\
                 pe[super_event]['spin1z'], pe[super_event]['spin2z'], pe[super_event]['lumd'])
            pe[super_event]['prior_pdf'] = prior_pdf
            pe[super_event]['redshift'] = dlum_to_z(pe[super_event]['lumd'])
            
            npe = len(prior_pdf)
            idxsel = np.random.choice(np.arange(npe), size = nsel, replace=False)
            for key in pe[super_event].keys():
                pe[super_event][key] = pe[super_event][key][idxsel]
    
    return pe

def get_pe_weights(mass1, mass2,  spin1z, spin2z, lumd):
    
    z = dlum_to_z(lumd)
    # Lalinference is uniform in masses and follows uniform in s or s_z in spins
    #Spins
    ps1z = - np.log(np.abs(spin1z)) / 2
    ps2z = - np.log(np.abs(spin2z)) / 2
    prior_pdf = ps1z * ps2z
    
    #Change from detector to source frame
    psrc = lumd ** 2 #uniform in L from uniform in V
    psrc *= (1 + z) ** 2#det mchirp to src
    psrc *= get_dLdz(z) #L to z
    prior_pdf *= psrc
     
    return prior_pdf
