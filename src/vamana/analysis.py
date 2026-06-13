import numpy as np
import vamana.priors as priors


class Analysis:
    def __init__(self, prior_file, data, ncomp, builder, likelihood_fnc, selection_fnc):
        """
        Core analysis engine for gravitational wave population inference
        using a Bayesian mixture model framework.

        Parameters
        ----------
        prior_file : str
            Path to the .prior configuration file specifying hyperparameter
            priors, distributions, and per-component flags.
        data : Data
            Container for observation runs, events, and injection sets,
            as defined in vamana.data.
        ncomp : int
            Number of mixture model components.
        builder : callable
            Maps the flattened theta array to a physical model payload
            dictionary, including derived parameters and normalisations.
            Signature: builder(theta, slices, fixed_params) -> dict
        likelihood_fnc : callable
            User-defined likelihood function.
            Signature: likelihood_fnc(payload, data, nobs, selection_fnc) -> float
        selection_fnc : callable
            User-defined selection effects function for computing the
            sensitive volume-time integral.
            Signature: selection_fnc(injections, ...) -> float

        Attributes
        ----------
        ncomp : int
            Number of mixture model components.
        data : Data
            The data object passed in at initialisation.
        builder : callable
            The builder function passed in at initialisation.
        likelihood_fnc : callable
            The likelihood function passed in at initialisation.
        selection_fnc : callable
            The selection function passed in at initialisation.
        priors_spec : dict
            Parsed prior specifications keyed by parameter name.
        param_names : list of str
            Ordered list of free parameter names for the flattened array.
        ndim : int
            Total number of free parameters (dimensionality of the problem).
        nobs : int
            Total number of observed events across all observation runs.
        slices : dict
            Index mapping from parameter names to positions in theta,
            used by the builder for efficient parameter extraction.
        name_to_idx : dict
            Mapping from parameter name to index in param_names.
        fixed_params : dict
            Dictionary of fixed parameter values keyed by parameter name.
        has_proposals : bool
            True if any parameter has a proposal function defined in the
            prior file.
        joint_proposals : dict
            Keyed by base parameter name, each value is a list of
            (j, name) tuples for parameters with joint proposals.
        sampler : object
            The sampler instance, set after calling run().
        results : object
            The results object from the sampler, set after calling run().

        Examples
        --------
        >>> from vamana.analysis import Analysis
        >>> from vamana.data import Data
        >>> from vamana.builders import mixture_builder
        >>> from vamana.likelihood import calculate_log_likelihood
        >>> from vamana.selection import get_vt
        >>> from vamana.samplers import run_dynesty, run_mcmc
        >>> data = Data()
        >>> data.add(
        ...     pattern="/data/GWTC-1/*bbh*",
        ...     reader="read_gwtc1",
        ...     label="O1O2",
        ...     nsamp=5000
        ... )
        >>> data.add_injections(
        ...     "/data/injections/o1o2.h5",
        ...     reader="read_injections_gwtc1",
        ...     label="O1O2",
        ...     DETSNR_THR=8.0,
        ...     NETSNR_THR=12.0
        ... )
        >>> data.check()
        >>> data.curate()
        >>> analysis = Analysis(
        ...     "model.prior", data, ncomp=10,
        ...     builder=mixture_builder,
        ...     likelihood_fnc=calculate_log_likelihood,
        ...     selection_fnc=get_vt
        ... )
        >>> analysis.run(run_dynesty, nlive=1000, ncpu=20, sample='rwalk')
        >>> analysis.run(run_mcmc, nsteps=500000, nburn=50000, thin=10)
        """
        self.ncomp          = ncomp
        self.data           = data
        self.builder        = builder
        self.likelihood_fnc = likelihood_fnc
        self.selection_fnc  = selection_fnc

        # 1. Parse prior file and setup dimensionality
        self.priors_spec = self._parse_prior(prior_file)
        self.param_names = self._get_param_names()
        self.ndim        = len(self.param_names)
        self.nobs        = self._count_observations()

        # 2. Setup index mapping for efficient parameter extraction in builder
        self.slices      = self._map_indices()
        self.name_to_idx = {name: i for i, name in enumerate(self.param_names)}

        # 3. Identify fixed parameters (not part of the sampling cube)
        self.fixed_params = {
            n: p['value'] for n, p in self.priors_spec.items()
            if p['dist'] == 'fixed'
        }

        # 4. Flag whether any parameter has a proposal function defined
        self.has_proposals = any(
            spec.get('proposal') is not None
            for spec in self.priors_spec.values()
        )

        # 5. Identify joint proposal parameters and their indices.
        #    Joint proposals (e.g. dirichlet_joint) propose all dimensions
        #    of a parameter group simultaneously in theta space.
        #    Stored as dict: base_name -> list of (j, param_name) tuples.
        self.joint_proposals = self._build_joint_proposals()

        self.sampler = None
        self.results = None

    def _count_observations(self):
        """
        Count the total number of observed events across all observation runs.

        Returns
        -------
        int
            Total number of observed events.
        """
        return sum(len(obsrun.events) for obsrun in self.data.obsruns.values())

    def _parse_prior(self, prior_file):
        """
        Parse a .prior file into a dictionary of prior specifications.

        The prior file format is (whitespace delimited):
            NAME  DEFINITION  DISTRIBUTION  PER_COMPONENT  ARG1  [ARG2 ...]
            [PROPOSAL  PROP_ARG1  ...]

        The proposal column is optional. It is identified as the first
        non-numeric, non-None value after the distribution args.

        Proposal types (identified by suffix):
            *_joint  — joint proposal for grouped parameters (theta space)
            *        — single parameter proposal (any space)

        Dynesty ignores all proposal columns.
        Lines starting with '#' are treated as comments and ignored.
        Fixed parameters use 'fixed' as the distribution.

        Parameters
        ----------
        prior_file : str
            Path to the .prior file.

        Returns
        -------
        dict
            Dictionary keyed by parameter name, each value is a dict with:
            - 'dist'          : str, distribution function name
            - 'per_component' : bool, whether to expand by ncomp
            - 'args'          : list, distribution arguments (non-fixed only)
            - 'value'         : float, fixed value (fixed parameters only)
            - 'proposal'      : str or None, full proposal function base name
            - 'is_joint'      : bool, True if proposal ends with '_joint'
            - 'proposal_args' : list, proposal function arguments
        """
        from pathlib import Path
        from importlib.resources import files

        # If prior_file is not an existing path, look inside the package
        if not Path(prior_file).exists():
            prior_file = files('vamana').joinpath(f'priors/{prior_file}')

        specs = {}
        with open(prior_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                name, dist = parts[0], parts[2].lower()
                spec = {
                    'dist'         : dist,
                    'per_component': parts[3].lower() == 'true'
                }

                if dist == 'fixed':
                    spec['value']         = float(parts[4])
                    spec['proposal']      = None
                    spec['is_joint']      = False
                    spec['proposal_args'] = []
                else:
                    # Find proposal column: first non-numeric, non-None value
                    proposal_idx = None
                    for j, part in enumerate(parts[4:], start=4):
                        try:
                            float(part)
                        except ValueError:
                            if part == 'None':
                                continue
                            proposal_idx = j
                            break

                    if proposal_idx is not None:
                        proposal_name = parts[proposal_idx]
                        spec['args']          = [
                            float(x) if x != 'None' else None
                            for x in parts[4:proposal_idx]
                        ]
                        spec['proposal']      = proposal_name
                        spec['is_joint']      = proposal_name.endswith('_joint')
                        spec['proposal_args'] = [
                            float(x) for x in parts[proposal_idx + 1:]
                            if x != 'None'
                        ]
                    else:
                        args = []
                        for part in parts[4:]:
                            try:
                                args.append(float(part) if part != 'None'
                                            else None)
                            except ValueError:
                                break
                        spec['args']          = args
                        spec['proposal']      = None
                        spec['is_joint']      = False
                        spec['proposal_args'] = []

                specs[name] = spec
        return specs

    def _get_param_names(self):
        """
        Build the ordered list of free parameter names for the flattened
        unit cube array passed to the sampler.

        Non-fixed parameters with per_component=True are expanded to
        ncomp entries with suffixes _0, _1, ..., _{ncomp-1}.
        Non-fixed parameters with per_component=False appear once.
        Fixed parameters are excluded entirely.

        Returns
        -------
        list of str
            Ordered list of free parameter names.
        """
        names = []
        for name, spec in self.priors_spec.items():
            if spec['dist'] != 'fixed':
                if spec['per_component']:
                    for i in range(self.ncomp):
                        names.append(f"{name}_{i}")
                else:
                    names.append(name)
        return names

    def _map_indices(self):
        """
        Create an index mapping from parameter names to positions in theta.

        For per-component parameters, stores a list of indices (one per
        component). For global parameters, stores a single index.
        Used by the builder for efficient parameter extraction.

        Returns
        -------
        dict
            Dictionary mapping parameter base names to indices or lists
            of indices in the flattened theta array.
        """
        slices = {}
        for name, spec in self.priors_spec.items():
            if spec['dist'] == 'fixed':
                continue
            if spec['per_component']:
                slices[name] = [
                    i for i, n in enumerate(self.param_names)
                    if n.startswith(f"{name}_")
                ]
            else:
                slices[name] = self.param_names.index(name)
        return slices

    def _build_joint_proposals(self):
        """
        Identify parameters with joint proposals (is_joint=True).
        Groups all parameters sharing the same joint proposal name.

        Returns
        -------
        dict
            Keyed by proposal name, each value is a list of
            (j, name) tuples in param_names order.
        """
        joint = {}
        for j, name in enumerate(self.param_names):
            base_name = name if name in self.priors_spec \
                        else '_'.join(name.split('_')[:-1])
            spec      = self.priors_spec[base_name]

            if spec.get('is_joint', False):
                proposal = spec['proposal']
                if proposal not in joint:
                    joint[proposal] = []
                joint[proposal].append((j, name))

        return joint

    def prior_transform(self, u):
        """
        Map a unit cube array to physical hyperparameter values using the
        inverse CDFs defined in priors.py.

        Always uses the distribution function (inverse CDF) for the mapping,
        regardless of whether a proposal function is defined. Proposal
        functions are only used by the sampler for generating proposals
        and computing MH corrections — they never affect the prior transform.

        This ensures dynesty and MCMC both use the same prior transform,
        and that stored posterior samples correctly reflect the prior.

        Parameters
        ----------
        u : np.ndarray
            Array of values in [0, 1] of length ndim.

        Returns
        -------
        np.ndarray
            theta array of physical hyperparameter values.
        """
        theta = np.zeros(len(u))

        for i, name in enumerate(self.param_names):
            base_name = name if name in self.priors_spec \
                        else '_'.join(name.split('_')[:-1])
            spec      = self.priors_spec[base_name]
            dist_fnc  = getattr(priors, spec['dist'])
            theta[i]  = dist_fnc(
                u[i],
                *spec.get('args', []),
                theta=theta,
                name_to_idx=self.name_to_idx,
                name=name,
                param_names=self.param_names
            )

        return theta

    def log_prior(self, u):
        """
        Log prior in the unit cube.

        Returns 0 if all values are in [0, 1], -inf otherwise.

        Parameters
        ----------
        u : np.ndarray
            Array of values in the unit cube of length ndim.

        Returns
        -------
        float
            0.0 if valid, -np.inf if out of bounds.
        """
        if np.any(u < 0) or np.any(u > 1):
            return -np.inf
        return 0.0

    def log_prior_theta(self, theta):
        """
        Log prior evaluated directly in physical theta space.

        Checks that all parameters are within their prior bounds.
        For uniform priors this is 0 if in bounds, -inf otherwise.
        For parameters with dynamic bounds (e.g. mu_m2 < mu_m1),
        checks the constraint directly.

        Parameters
        ----------
        theta : np.ndarray
            Array of physical hyperparameter values of length ndim.

        Returns
        -------
        float
            0.0 if valid, -np.inf if any parameter is out of bounds.
        """
        for j, name in enumerate(self.param_names):
            base_name = name if name in self.priors_spec \
                        else '_'.join(name.split('_')[:-1])
            spec      = self.priors_spec[base_name]
            dist      = spec['dist']
            args      = spec.get('args', [])

            if dist == 'fixed':
                continue

            val = theta[j]

            if dist in ('uniform', 'uniform_in_log', 'm1_for_uniform_m1m2',
                        'm1_for_uniform_m1m2_ordered'):
                lo = args[0] if len(args) > 0 and args[0] is not None else None
                hi = args[1] if len(args) > 1 and args[1] is not None else None
                if lo is not None and val < lo: return -np.inf
                if hi is not None and val > hi: return -np.inf

            elif dist == 'uniform_min_m2_to_m1':
                lo       = args[0] if len(args) > 0 and args[0] is not None else 0.0
                comp_idx = name.split('_')[-1]
                m1_idx   = self.param_names.index(f'mu_m1_{comp_idx}')
                hi       = theta[m1_idx]
                if val < lo or val > hi: return -np.inf

            elif dist == 'piecewise_uniform_logarithmic':
                b = args[1] if len(args) > 1 and args[1] is not None else None
                if b is not None and abs(val) > b: return -np.inf

            elif dist == 'uniform_kcomp':
                half_width = args[0] if len(args) > 0 else 2.0
                kpop_idx   = self.param_names.index('kpop')
                kpop       = theta[kpop_idx]
                if val < kpop - half_width or val > kpop + half_width:
                    return -np.inf

            elif dist == 'dirichlet':
                if val < 0 or val > 1: return -np.inf

        return 0.0

    def log_posterior_theta(self, theta):
        """
        Log posterior evaluated directly in physical theta space.
        Used by MCMC samplers that work in theta space.

        Parameters
        ----------
        theta : np.ndarray

        Returns
        -------
        float
            Log posterior. Positive value signals neff failure (skip step).
            -inf if out of prior bounds.
        """
        if len(theta) != self.ndim:
            raise ValueError(
                f"theta has wrong shape: expected {self.ndim}, "
                f"got {len(theta)}. "
                f"Check param_names for expected order: {self.param_names}"
            )
        lp = self.log_prior_theta(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def log_likelihood(self, theta):
        """
        Likelihood wrapper for the sampler.

        Maps theta to a model payload via the builder, then calls the
        user-defined likelihood function. Catches numerical instabilities
        and returns a positive value to signal the sampler to skip the step.

        Parameters
        ----------
        theta : np.ndarray
            Array of physical hyperparameter values of length ndim.

        Returns
        -------
        float
            Log likelihood value. A positive value signals neff failure
            or numerical instability — the sampler should skip the step.
        """
        try:
            payload = self.builder(theta, self.slices, self.fixed_params)
            return self.likelihood_fnc(
                payload=payload,
                data=self.data,
                nobs=self.nobs,
                selection_fnc=self.selection_fnc
            )
        except (ValueError, np.linalg.LinAlgError):
            return 1.0  # positive signals skip

    def log_posterior(self, u):
        """
        Log posterior in the unit cube.

        Parameters
        ----------
        u : np.ndarray
            Array of values in the unit cube of length ndim.

        Returns
        -------
        float
            Log posterior value. A positive value signals neff failure
            or numerical instability (skip step). -inf if out of bounds.
        """
        lp = self.log_prior(u)
        if not np.isfinite(lp):
            return -np.inf
        theta = self.prior_transform(u)
        return lp + self.log_likelihood(theta)

    def run(self, sampler_fn, **kwargs):
        """
        Run the analysis using the specified sampler function.

        Parameters
        ----------
        sampler_fn : callable
            Sampler function from vamana.samplers.
            Signature: sampler_fn(analysis, **kwargs) -> results
        **kwargs
            Additional keyword arguments passed to the sampler function.

        Returns
        -------
        object
            Results object returned by the sampler function.

        Examples
        --------
        >>> from vamana.samplers import run_dynesty, run_mcmc
        >>> analysis.run(run_dynesty, nlive=1000, ncpu=20, sample='rwalk')
        >>> analysis.run(run_mcmc, nsteps=500000, nburn=50000, thin=10)
        """
        self.results = sampler_fn(self, **kwargs)
        return self.results

    def get_samples(self):
        """
        Extract equally weighted posterior samples.

        For dynesty results, resamples to equal weights.
        For MCMC results, returns samples directly.

        Returns
        -------
        dict
            Dictionary keyed by parameter name, each value is an array of
            posterior samples in physical space.

        Raises
        ------
        ValueError
            If run() has not been called yet.

        Examples
        --------
        >>> samples = analysis.get_samples()
        >>> samples['mu_m1_0']
        array([...])
        """
        if self.results is None:
            raise ValueError("No results found. Run analysis.run() first.")

        # MCMC results
        if isinstance(self.results, dict) and 'theta_samples' in self.results:
            return {
                name: self.results['theta_samples'][:, i]
                for i, name in enumerate(self.param_names)
            }

        # Dynesty results
        import dynesty.utils as dyfunc
        weights       = np.exp(self.results.logwt - self.results.logz[-1])
        equal_samples = dyfunc.resample_equal(self.results.samples, weights)
        return {name: equal_samples[:, i]
                for i, name in enumerate(self.param_names)}