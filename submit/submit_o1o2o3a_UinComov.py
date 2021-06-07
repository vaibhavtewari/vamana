import os, sys
from pathos.parallel import stats
from pathos.parallel import ParallelPool as Pool
pool = Pool()

sys.path.append("../vamana/")
sys.path.append("../vamana/analysis/")
import post_process

def host(id):
    
    import glob
    
    import numpy as np
    import functions, read_pe_samples
    import o1o2o3a_UinComov as analysis
    
    # Read data
    ifar_thr = 5.0
    nsamp = 7000
    injections, pe = {}, {}
    pe_dir = 'gw_data/o1o2_ifar5_phenompv2/*.hdf5'
    pe['o1o2'] = read_pe_samples.read_pesamples(pe_dir, nsamp)
    pe_dir = 'gw_data/o3a_ifar5_phenompv2/*.hdf5'
    pe['o3a'] = read_pe_samples.read_pesamples(pe_dir, nsamp)

    fname = 'gw_data/o1o2_bbhpop_siminj.hdf'
    injections['o1o2'] = functions.read_injections_o1o2(fname)
    fname = 'gw_data/o3a_bbhpop_inj_info.hdf'
    injections['o3a'] = functions.read_injections_o3(fname, ifar_thr, 'aligned')
    
    args_sampler, args_ppd = analysis.define_args(pe, injections)
    output = args_ppd['output']
    
    fout = 'results/' + analysis.__name__ + '_ng' + str(args_sampler['ngauss']) 
    fout += '_AD' + str(args_sampler['AD_thr_mch']) + '_ifar' + str(ifar_thr) 
    fout += '_pv2_ak1.hdf5'
        
    if len( glob.glob(output + '*') ) == 0:
    
        data_analysis = {}
        data_analysis['pe'] = pe
        data_analysis['injections'] = injections
        data_analysis['analysis'] = analysis
    
        results = functions.function_gauss(data_analysis)
    
        return fout, args_sampler, args_ppd
    else:
        return fout, args_sampler, args_ppd

pool.ncpus = 10
pool.servers = ('localhost:0',)
result = pool.map(host, range(10))
fout, args_sampler, args_ppd = result[0]
print ('Saved to file:', fout)
if os.path.exists(fout):
    print ('File already exist! Proceed ?')
    choice = input().lower()
    if choice == 'yes' or choice == 'y':
        post_process.gather_files(fout, args_sampler, args_ppd)
else:
    post_process.gather_files(fout, args_sampler, args_ppd)