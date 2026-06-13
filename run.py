import os
import sys
import glob
import numpy as np
import scipy.stats as ss
from vamana.analysis import Analysis
from vamana.data import Data
from vamana.curate import get_data_for_fitting
from vamana.builders import mixture_builder
from vamana.likelihood import calculate_log_likelihood
from vamana.selection import get_vt
from vamana.samplers import run_mcmc_uniform_step, run_mcmc
from vamana.utils import combine_chains

# ---- Chain ID from condor ----
chain_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# ---- Load data ----
data = Data(curate_fn=get_data_for_fitting)

IFAR_thr = 1.0
nsamp = 2000

data.add(pattern='./gw_data/pe/*o1o2*BBH*.hdf5', reader="read_pe_gwtc1", label="GWTC1", nsamp=nsamp)
fname = './gw_data/injections/o1+o2-bbh-IMRPhenomXPHMpseudoFourPN.hdf5'
data.add_injections(fname, reader='read_injections_gwtc1', label='GWTC1', DETSNR_THR=3.0, NETSNR_THR=9.0)

data.add(pattern='./gw_data/pe/*o3a*BBH*.hdf5', reader="read_pe_o3", label="GWTC2", nsamp=nsamp)
fname = './gw_data/injections/endo3_bbhpop-LIGO-T2100113-v9-1238166018-15843600.hdf5'
data.add_injections(fname, reader='read_injections_o3', IFAR_thr=IFAR_thr, label="GWTC2")

data.add(pattern='./gw_data/pe/*o3b*BBH*.hdf5', reader="read_pe_o3", label="GWTC3", nsamp=nsamp)
fname = './gw_data/injections/endo3_bbhpop-LIGO-T2100113-v9-1256655642-12905976.hdf5'
data.add_injections(fname, reader='read_injections_o3', IFAR_thr=IFAR_thr, label="GWTC3")

# ---- Validate ----
data.check()

# ---- Curate ----
data.curate()

#initialise starting u(speeds intitial sampling)
ncomp = 10
min_mass, max_mass = 6.0, 75.0
mu_m1 = np.random.uniform(min_mass, max_mass, ncomp)
#weights
w = 1/mu_m1
w /= w.sum()

theta_start = w
theta_start = np.append(theta_start, mu_m1)

#mu_m2
mu_m2 = np.maximum(min_mass, mu_m1 * np.random.uniform(0.7, 1.0, ncomp))
theta_start = np.append(theta_start, mu_m2)

# sigma_over_mu_m1_pow_beta and sigma_over_mu_m2_pow_beta
theta_start = np.append(theta_start, np.random.uniform(0.10, 0.14, ncomp))
theta_start = np.append(theta_start, np.random.uniform(0.10, 0.14, ncomp))

# rho_m1m2
theta_start = np.append(theta_start, np.random.uniform(-0.5, 0.5, ncomp))

# mu_sz and sigma_sz
theta_start = np.append(theta_start, np.random.uniform(-0.5, 0.5, ncomp))
theta_start = np.append(theta_start, np.random.uniform(0.15, 0.3, ncomp))

# k_pop and k_comp
k = np.random.uniform(-1, 1)
theta_start = np.append(theta_start, k)
theta_start = np.append(theta_start, np.random.uniform(k-1, k+1, ncomp))

#rate
theta_start = np.append(theta_start, np.random.uniform(5, 50))



# ---- Run analysis ----
analysis = Analysis(
    "model_GWTC3.prior", data, ncomp=10,
    builder=mixture_builder,
    likelihood_fnc=calculate_log_likelihood,
    selection_fnc=get_vt
)
        
analysis.run(run_mcmc_uniform_step, nsteps=80000,
                 nburn=40000, thin=1000,
                 delta_u = 0.03,
                 progress_interval=10,
                 theta_start=theta_start,
                 chain_id=chain_id,
                 output='temp/GWTC3')