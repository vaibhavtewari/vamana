#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob, sys, os
import numpy as np

from conversions import *
import gnobs, models, functions
import read_pe_samples
import post_process

sys.path.append("analysis/")

case = 1

pe_dir, inj_file = {}, {}
pe_dir['o1o2'] = 'gw_data/pe/*o1o2*BBH*.hdf5'
pe_dir['o3a'] = 'gw_data/pe/*o3a*BBH*.hdf5'
pe_dir['o3b'] = 'gw_data/pe/*o3b*BBH*.hdf5'
pe_GW190814 = 'gw_data/pe/S190814bv*.hdf5'

inj_file['o1o2'] = 'gw_data/injections/o1o2_bbhpop_siminj.hdf'
inj_file['o3a'] = 'gw_data/injections/endo3_bbhpop-LIGO-T2100113-v9-1238166018-15843600.hdf5'
inj_file['o3b'] = 'gw_data/injections/endo3_bbhpop-LIGO-T2100113-v9-1256655642-12905976.hdf5'

nsamp = 5000
def read_data(obsruns, ifar_thr):
    injections, pe = {}, {}
    for obsrun in obsruns:
        pe[obsrun] = read_pe_samples.read_pesamples(pe_dir[obsrun], ifar_thr, nsamp)

        if obsrun == 'o1o2':
            injections[obsrun] = functions.read_injections_o1o2(inj_file[obsrun])
        else:
            injections[obsrun] = functions.read_injections_o3(inj_file[obsrun], ifar_thr)
            
    return pe, injections
            
if case == 1:
    import o1o2o3_sfr_1pz_ifar_1peryear as analysis
    pe, injections = read_data(['o1o2', 'o3a', 'o3b'], 1.0)
if case == 2:
    import o1o2o3_sfr_1pz_ifar_1peryear_mixture as analysis
    pe, injections = read_data(['o1o2', 'o3a', 'o3b'], 1.0)
if case == 3:
    import o1o2o3_sfr_1pz_ifar_1peryear_mixture_with_GW190814 as analysis
    pe, injections = read_data(['o1o2', 'o3a', 'o3b'], 1.0)
    pe_gw190814 = read_pe_samples.read_pesamples(pe_GW190814, 1.0, nsamp)
    key = list(pe_gw190814.keys())[0]
    pe['o3a'][key] = pe_gw190814[key]

data_analysis = {}
data_analysis['pe'] = pe
data_analysis['injections'] = injections
data_analysis['analysis'] = analysis

args_sampler, args_ppd = analysis.define_args(pe, injections)
output = args_ppd['output']

analysis_name = analysis.__name__
print (analysis_name)
fout = 'results/' + analysis_name + '_ng' + str(args_sampler['ngauss']) +'.hdf5'
files = np.sort(glob.glob(output + '/*_' + analysis_name + '_*.hdf5'))
try:
    flen = np.array([len(fn.split('_')) for fn in files])
    files = files[flen == len(analysis_name.split('_')) + 3]
    nfiles = len(files)
except:
    results = functions.function_gauss(data_analysis)
if nfiles > 0:
    print ('Combining existing files. Saving to file:', fout)
    print ('One of the files to combine:', files[np.random.randint(nfiles)])
    print ('Number of Files', nfiles)
    if os.path.exists(fout):
        print ('File already exist! Proceed ?')
        choice = input().lower()
        if choice == 'yes' or choice == 'y':
            post_process.gather_files(fout, analysis_name, args_sampler, args_ppd, files)
    else:
        post_process.gather_files(fout, analysis_name, args_sampler, args_ppd, files)
else:
    results = functions.function_gauss(data_analysis)