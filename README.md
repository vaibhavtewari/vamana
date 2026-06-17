# Vamana

Vamana is a Python package for Bayesian population inference of binary black hole (BBH) mergers using gravitational wave observations. It models the BBH population as a Gaussian mixture over component masses and aligned spins, with full correction for detector selection effects and merger rate estimation.

## Methodology

The methodology is described in:

- Tiwari, V. (2021). *VAMANA: modeling binary black hole population with minimal assumptions.* Classical and Quantum Gravity, 38(15), 155007. [doi:10.1088/1361-6382/ac0b54](https://doi.org/10.1088/1361-6382/ac0b54)
- Tiwari, V. (2018). *Estimation of the sensitive volume for gravitational-wave source populations using weighted Monte Carlo integration.* Classical and Quantum Gravity, 35(14), 145009. [doi:10.1088/1361-6382/aac89d](https://doi.org/10.1088/1361-6382/aac89d)

Scientific results obtained using Vamana are presented in:

- Tiwari, V. & Fairhurst, S. (2021). *The emergence of structure in the binary black hole mass distribution.* ApJL, 913(2), L19. [doi:10.3847/2041-8213/abfbe7](https://doi.org/10.3847/2041-8213/abfbe7)
- Tiwari, V. (2022). *Exploring features in the binary black hole population.* ApJ, 928(2), 155. [doi:10.3847/1538-4357/ac589a](https://doi.org/10.3847/1538-4357/ac589a)
- Tiwari, V. (2023). *What's in a binary black hole's mass parameter?* MNRAS, 527(1), 298. [doi:10.1093/mnras/stad3155](https://doi.org/10.1093/mnras/stad3155)
- Tiwari, V. (2025). *Examining the gap in the chirp mass distribution of binary black holes.* ApJ, 995, 177. [doi:10.3847/1538-4357/ae2260](https://doi.org/10.3847/1538-4357/ae2260)
- Tiwari, V. (2025). *Population of binary black holes inferred from one hundred and fifty gravitational wave signals.* arXiv:2510.25579.
- Tiwari, V. (2026). *The Chirp-Mass Ladder: A New Rung Emerges.* arXiv:2606.18081.

## Installation

```bash
pip install vamana
```

To use the nested sampler (dynesty):

```bash
pip install vamana[nested]
```

## Overview

Vamana models the BBH population as a mixture of weighted Gaussians over primary mass, secondary mass, and aligned spin components. The number of mixture components, prior specification, and likelihood function are all user-configurable.

### Key components

**`Data`** — loads and curates parameter estimation samples and injection campaigns across multiple observation runs.

**`Analysis`** — combines data, prior specification, builder, and likelihood into a single object for sampling.

**`run_mcmc_uniform_step`** — Metropolis-Hastings sampler with uniform CDF step proposals, adaptive step size, and joint proposals for correlated parameters.

**`run_dynesty`** — nested sampling via dynesty.

**Prior files** — plain-text prior specification files declaring parameter distributions, bounds, and proposal functions. Vamana ships with two ready-to-use prior files:

- `model_GWTC3.prior` — prior configuration for GWTC-1 through GWTC-3
- `model_GWTC5.prior` — prior configuration for GWTC-1 through GWTC-5

These are automatically found by name after `pip install vamana`. Users can also supply their own prior file by passing a full path.

### Prior file format

```
# NAME    DEFINITION    DISTRIBUTION    Per_component    arg1    arg2    Proposal    pargs
weights   ...           dirichlet       True             0.0     1.0     dirichlet_joint
mu_m1     ...           m1_for_uniform_m1m2  True        6.0     75.0    mass_powerlaw_joint  -1.1
...
rate      ...           uniform_in_log  False            1.0     100.0
```

Parameters with no proposal function declared are moved using their distribution's CDF-step draw function in `run_mcmc_uniform_step`, or via adaptive Metropolis in `run_mcmc`.

### Data loading

```python
from vamana import Data
from vamana.curate import get_data_for_fitting

data = Data(curate_fn=get_data_for_fitting)

# GWTC-1 (O1+O2)
data.add(pattern='/path/to/pe/*o1o2*BBH*.hdf5', reader='read_pe_gwtc1',
         label='GWTC1', nsamp=2000)
data.add_injections('/path/to/injections/o1o2.hdf5',
                    reader='read_injections_gwtc1', label='GWTC1',
                    DETSNR_THR=3.0, NETSNR_THR=9.0)

# GWTC-2 (O3a)
data.add(pattern='/path/to/pe/*o3a*BBH*.hdf5', reader='read_pe_o3',
         label='GWTC2', nsamp=2000)
data.add_injections('/path/to/injections/o3a.hdf5',
                    reader='read_injections_o3', label='GWTC2', IFAR_thr=1.0)

# GWTC-3 (O3b)
data.add(pattern='/path/to/pe/*o3b*BBH*.hdf5', reader='read_pe_o3',
         label='GWTC3', nsamp=2000)
data.add_injections('/path/to/injections/o3b.hdf5',
                    reader='read_injections_o3', label='GWTC3', IFAR_thr=1.0)

data.check()
data.curate()
```

### Running an analysis

```python
from vamana.analysis import Analysis
from vamana.builders import mixture_builder
from vamana.likelihood import calculate_log_likelihood
from vamana.selection import get_vt
from vamana.samplers import run_mcmc_uniform_step

analysis = Analysis(
    'model_GWTC3.prior', data, ncomp=10,
    builder=mixture_builder,
    likelihood_fnc=calculate_log_likelihood,
    selection_fnc=get_vt
)

analysis.run(
    run_mcmc_uniform_step,
    nsteps=200000, nburn=100000, thin=1000,
    delta_u=0.015, chain_id=0,
    output='results/run'
)
```

This saves `results/run_chain0.h5`. For multiple chains, run the same script with different `chain_id` values — on a cluster via condor or slurm, pass the chain ID as a command-line argument:

```python
import sys
chain_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
analysis.run(..., chain_id=chain_id, output='results/run')
```

### Checking convergence and combining chains

Once all chains are complete, check convergence before combining:

```python
from vamana.utils import convergence_report, combine_chains

# Gelman-Rubin R-hat and autocorrelation length across all chains
convergence_report('results/run_chain*.h5')
```

Output looks like:
```
==================================================
Convergence Report
==================================================
Chains:          8
Samples/chain:   100
Total samples:   800
==================================================

Gelman-Rubin R-hat:
  Max R-hat:     1.023
  Mean R-hat:    1.008
  Converged:     True
  Worst param:   kpop (R-hat=1.023)

Autocorrelation:
  Mean ACL:      12.3
  Max ACL:       28.1 (mu_m1_3)
  Min ESS:       284
  Suggested thin: 29
==================================================
```

If R-hat < 1.1 and ESS is sufficient, combine:

```python
combine_chains('results/run_chain*.h5', output='results/combined.h5')
```

### Loading posterior samples

```python
from vamana.utils import load_samples

samples = load_samples('results/combined.h5')
samples['mu_m1'].shape   # (nsamples, ncomp)
samples['rate'].shape    # (nsamples,)
```

## Requirements

- Python >= 3.9
- numpy >= 1.21
- scipy >= 1.7
- astropy >= 5.0
- h5py >= 3.0
- dynesty >= 2.0 (optional, for nested sampling)

## Licence

MIT
