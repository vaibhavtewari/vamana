import numpy as np
import glob
import h5py


def autocorrelation_length(chain, max_lag=None, threshold=0.05):
    """
    Estimate the integrated autocorrelation length (ACL) for a 1D chain.

    Parameters
    ----------
    chain : np.ndarray of shape (nsamples,)
        1D array of samples from a single parameter.
    max_lag : int, optional
        Maximum lag to consider. Default is None, which uses n // 2.
    threshold : float, optional
        Stop summing when autocorrelation drops below this value.
        Default is 0.05.

    Returns
    -------
    float
        Integrated autocorrelation length.

    Examples
    --------
    >>> acl = autocorrelation_length(samples[:, 0])
    >>> print(f"ACL: {acl:.0f}")
    """
    n    = len(chain)
    mean = np.mean(chain)
    var  = np.var(chain)

    if var == 0:
        return 1.0

    max_lag = max_lag if max_lag else n // 2
    acl     = 1.0

    for lag in range(1, max_lag):
        rho = np.mean((chain[:-lag] - mean) * (chain[lag:] - mean)) / var
        if rho < threshold:
            break
        acl += 2 * rho

    return acl


def compute_acl(samples, param_names=None, threshold=0.05):
    """
    Estimate the integrated autocorrelation length for all parameters.

    Parameters
    ----------
    samples : np.ndarray of shape (nsamples, ndim)
        Posterior samples, one column per parameter.
    param_names : list of str, optional
        Parameter names for reporting. Default is None.
    threshold : float, optional
        Stop summing when autocorrelation drops below this value.
        Default is 0.05.

    Returns
    -------
    dict
        Dictionary containing acl, mean_acl, max_acl, max_acl_param,
        n_effective, and min_n_effective.

    Examples
    --------
    >>> acl_stats = compute_acl(samples, param_names=analysis.param_names)
    >>> print(f"Suggested thin: {int(np.ceil(acl_stats['max_acl']))}")
    """
    nsamples, ndim = samples.shape
    acl            = np.array([
        autocorrelation_length(samples[:, i], threshold=threshold)
        for i in range(ndim)
    ])
    n_effective   = nsamples / acl
    max_acl_idx   = np.argmax(acl)
    max_acl_param = param_names[max_acl_idx] if param_names else str(max_acl_idx)

    print(f"Mean ACL:        {acl.mean():.1f}")
    print(f"Max ACL:         {acl.max():.1f} ({max_acl_param})")
    print(f"Min ESS:         {n_effective.min():.0f}")
    print(f"Suggested thin:  {int(np.ceil(acl.max()))}")

    return {
        'acl'            : acl,
        'mean_acl'       : acl.mean(),
        'max_acl'        : acl.max(),
        'max_acl_param'  : max_acl_param,
        'n_effective'    : n_effective,
        'min_n_effective': n_effective.min()
    }


def _read_prior_spec(f):
    """
    Read prior specification from an open HDF5 file handle.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle.

    Returns
    -------
    dict
        Parsed prior specification keyed by parameter name.
    """
    prior_spec = {}
    for param in f['prior'].keys():
        pg   = f[f'prior/{param}']
        spec = {}
        for k, v in pg.attrs.items():
            spec[k] = None if v == 'None' else v
        for k in pg.keys():
            arr = pg[k][:]
            spec[k] = [None if np.isnan(x) else x for x in arr]
        prior_spec[param] = spec
    return prior_spec


def _check_priors_match(ref_spec, chain_spec, filepath):
    """
    Check that two prior specs match and report mismatches.

    Parameters
    ----------
    ref_spec : dict
        Reference prior spec from first chain.
    chain_spec : dict
        Prior spec from chain being checked.
    filepath : str
        Path to chain file for error messages.

    Raises
    ------
    ValueError
        If prior specs don't match, with detailed diagnostic information.
    """
    mismatches = []

    ref_keys   = set(ref_spec.keys())
    chain_keys = set(chain_spec.keys())

    if ref_keys != chain_keys:
        missing = ref_keys - chain_keys
        extra   = chain_keys - ref_keys
        if missing:
            mismatches.append(f"  Missing parameters: {missing}")
        if extra:
            mismatches.append(f"  Extra parameters:   {extra}")

    for name in ref_keys & chain_keys:
        rs = ref_spec[name]
        cs = chain_spec[name]

        if rs.get('dist') != cs.get('dist'):
            mismatches.append(
                f"  '{name}': dist mismatch "
                f"(ref='{rs.get('dist')}', chain='{cs.get('dist')}')"
            )
        if rs.get('per_component') != cs.get('per_component'):
            mismatches.append(
                f"  '{name}': per_component mismatch "
                f"(ref={rs.get('per_component')}, "
                f"chain={cs.get('per_component')})"
            )
        if 'value' in rs and 'value' in cs:
            if not np.isclose(float(rs['value']), float(cs['value'])):
                mismatches.append(
                    f"  '{name}': value mismatch "
                    f"(ref={rs['value']}, chain={cs['value']})"
                )
        if 'args' in rs and 'args' in cs:
            ref_args   = [a for a in rs['args'] if a is not None]
            chain_args = [a for a in cs['args'] if a is not None]
            if len(ref_args) != len(chain_args):
                mismatches.append(
                    f"  '{name}': args length mismatch "
                    f"(ref={ref_args}, chain={chain_args})"
                )
            elif ref_args and not np.allclose(ref_args, chain_args):
                mismatches.append(
                    f"  '{name}': args mismatch "
                    f"(ref={ref_args}, chain={chain_args})"
                )

    if mismatches:
        raise ValueError(
            f"\nPrior mismatch in {filepath}:\n" + "\n".join(mismatches)
        )


def load_chain(filepath):
    """
    Load a single HDF5 chain file.

    Parameters
    ----------
    filepath : str
        Path to the chain HDF5 file saved by run_mcmc or run_mcmc_uniform.

    Returns
    -------
    tuple
        (theta_samples, log_post, param_names, prior_spec,
         fixed_params, metadata) where:
        - theta_samples : np.ndarray of shape (nsamples, ndim)
        - log_post      : np.ndarray of shape (nsamples,)
        - param_names   : list of str, in prior file order
        - prior_spec    : dict, full prior specification
        - fixed_params  : dict, fixed parameter values
        - metadata      : dict, chain metadata

    Examples
    --------
    >>> theta, log_post, param_names, prior_spec, fixed, meta = load_chain('chain0.h5')
    """
    with h5py.File(filepath, 'r') as f:
        theta_samples = f['samples/theta'][:]
        log_post      = f['samples/log_post'][:]
        param_names   = [n.decode() for n in f['param_names'][:]]
        prior_spec    = _read_prior_spec(f)
        fixed_params  = {k: float(v)
                         for k, v in f['fixed_params'].attrs.items()} \
                        if 'fixed_params' in f else {}
        metadata      = dict(f['metadata'].attrs)

    return theta_samples, log_post, param_names, prior_spec, fixed_params, metadata


def combine_chains(pattern, output='combined_samples.h5'):
    """
    Load and combine posterior samples from multiple HDF5 chain files.

    All information (param_names, prior spec, fixed params, metadata) is
    read directly from the chain files — no user input needed. Validates
    consistency of param_names and prior specs across chains before combining.

    Per-component parameters are grouped under their base name as a 2D
    array of shape (nsamples, ncomp), in prior file order.

    Parameters
    ----------
    pattern : str
        Glob pattern matching chain HDF5 files e.g. 'GWTC3_chain*.h5'.
    output : str, optional
        Path to output HDF5 file. Default is 'combined_samples.h5'.

    Returns
    -------
    str
        Path to the saved HDF5 file.

    Raises
    ------
    FileNotFoundError
        If no files match the pattern.
    ValueError
        If prior specifications or param_names differ across chains.

    Examples
    --------
    >>> combine_chains('GWTC3_chain*.h5', output='GWTC3_combined.h5')
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching: {pattern}")

    all_theta         = []
    all_log_post      = []
    prior_specs       = []
    fixed_params_list = []
    param_names_list  = []
    metadatas         = []

    for filepath in files:
        theta, log_post, param_names, prior_spec, fixed_params, metadata = \
            load_chain(filepath)
        all_theta.append(theta)
        all_log_post.append(log_post)
        prior_specs.append(prior_spec)
        fixed_params_list.append(fixed_params)
        param_names_list.append(param_names)
        metadatas.append(metadata)

    # --- Validate consistency across chains ---
    ref_param_names = param_names_list[0]
    ref_spec        = prior_specs[0]
    ref_fixed       = fixed_params_list[0]

    for i, (pn, ps, fp) in enumerate(
        zip(param_names_list[1:], prior_specs[1:], fixed_params_list[1:]),
        start=1
    ):
        if pn != ref_param_names:
            raise ValueError(
                f"param_names mismatch in {files[i]}:\n"
                f"  ref:   {ref_param_names}\n"
                f"  chain: {pn}"
            )
        _check_priors_match(ref_spec, ps, files[i])
        for k in ref_fixed:
            if k not in fp or not np.isclose(ref_fixed[k], fp[k]):
                raise ValueError(
                    f"Fixed parameter '{k}' mismatch in {files[i]}: "
                    f"ref={ref_fixed[k]}, chain={fp.get(k)}"
                )

    print(f"Consistency check passed across {len(files)} chains")

    theta_samples = np.concatenate(all_theta)
    log_post      = np.concatenate(all_log_post)
    param_names   = ref_param_names
    ndim          = theta_samples.shape[1]
    ncomp         = max(
        int(name.split('_')[-1]) + 1
        for name in param_names
        if name.split('_')[-1].isdigit()
    )

    print(f"Loaded {len(files)} chains, total samples: {len(theta_samples)}")

    # --- Group per-component parameters in prior file order ---
    groups = {}
    for i, name in enumerate(param_names):
        parts = name.split('_')
        try:
            int(parts[-1])
            base_name        = '_'.join(parts[:-1])
            is_per_component = True
        except ValueError:
            base_name        = name
            is_per_component = False

        if is_per_component:
            if base_name not in groups:
                groups[base_name] = {}
            groups[base_name][int(parts[-1])] = theta_samples[:, i]
        else:
            groups[base_name] = theta_samples[:, i]

    # --- Save to combined HDF5 ---
    with h5py.File(output, 'w') as f:

        # Posterior samples in prior file order
        sample_grp = f.create_group('samples')
        for key, value in groups.items():
            if isinstance(value, dict):
                arr = np.stack(
                    [value[k] for k in sorted(value.keys())], axis=1
                )
                sample_grp.create_dataset(key, data=arr)
            else:
                sample_grp.create_dataset(key, data=value)
        sample_grp.create_dataset('log_post', data=log_post)

        # param_names in prior file order
        f.create_dataset('param_names',
                         data=np.array(param_names, dtype='S'))

        # Full prior spec
        prior_grp = f.create_group('prior')
        for param, spec in ref_spec.items():
            pg = prior_grp.create_group(param)
            for k, v in spec.items():
                if v is None:
                    pg.attrs[k] = 'None'
                elif isinstance(v, list):
                    cleaned = [x if x is not None else float('nan') for x in v]
                    pg.create_dataset(k, data=np.array(cleaned, dtype=float))
                else:
                    pg.attrs[k] = v

        # Fixed parameters
        fixed_grp = f.create_group('fixed_params')
        for k, v in ref_fixed.items():
            fixed_grp.attrs[k] = v

        # Metadata — global summary and per-chain details
        meta_grp = f.create_group('metadata')
        meta_grp.attrs['n_chains']      = len(files)
        meta_grp.attrs['total_samples'] = len(theta_samples)
        meta_grp.attrs['ndim']          = ndim
        meta_grp.attrs['ncomp']         = ncomp
        for i, (meta, fname) in enumerate(zip(metadatas, files)):
            chain_grp = meta_grp.create_group(f'chain_{i}')
            chain_grp.attrs['file'] = fname
            for k, v in meta.items():
                chain_grp.attrs[k] = v

    print(f"Saved to {output}")
    print(f"Free parameters:  {list(groups.keys())}")
    print(f"Fixed parameters: {list(ref_fixed.keys())}")

    return output


def load_samples(filepath, keys=None):
    """
    Load posterior samples from a combined HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file saved by combine_chains.
    keys : list of str, optional
        List of parameter names to load. Default is None (load all).

    Returns
    -------
    dict
        Dictionary keyed by parameter name. Per-component parameters
        have shape (nsamples, ncomp), global parameters have shape
        (nsamples,).

    Examples
    --------
    >>> samples = load_samples('GWTC3_combined.h5')
    >>> samples['mu_m1'].shape
    (50000, 10)
    >>> samples['rate'].shape
    (50000,)
    """
    with h5py.File(filepath, 'r') as f:
        sample_keys = list(f['samples'].keys())
        if keys is None:
            keys = sample_keys
        return {key: f[f'samples/{key}'][:]
                for key in keys if key in sample_keys}


def load_prior_spec(filepath):
    """
    Load the prior specification and fixed parameters from a chain
    or combined HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to an HDF5 file saved by run_mcmc, run_mcmc_uniform,
        or combine_chains.

    Returns
    -------
    tuple
        (prior_spec, fixed_params) where:
        - prior_spec   : dict, full prior specification
        - fixed_params : dict, fixed parameter values

    Examples
    --------
    >>> prior_spec, fixed_params = load_prior_spec('GWTC3_chain0.h5')
    >>> print(prior_spec['mu_m1'])
    >>> print(fixed_params)
    """
    with h5py.File(filepath, 'r') as f:
        prior_spec   = _read_prior_spec(f)
        fixed_params = {k: float(v)
                        for k, v in f['fixed_params'].attrs.items()} \
                       if 'fixed_params' in f else {}
    return prior_spec, fixed_params


def load_param_names(filepath):
    """
    Load parameter names from a chain or combined HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to an HDF5 file saved by run_mcmc, run_mcmc_uniform,
        or combine_chains.

    Returns
    -------
    list of str
        Parameter names in prior file order.

    Examples
    --------
    >>> param_names = load_param_names('GWTC3_chain0.h5')
    """
    with h5py.File(filepath, 'r') as f:
        return [n.decode() for n in f['param_names'][:]]


def gelman_rubin(pattern):
    """
    Compute the Gelman-Rubin convergence diagnostic (R-hat) for multiple
    HDF5 chain files.

    R-hat close to 1 indicates convergence. Values > 1.1 suggest the
    chains have not converged.

    Parameters
    ----------
    pattern : str
        Glob pattern matching chain HDF5 files e.g. 'GWTC3_chain*.h5'.

    Returns
    -------
    np.ndarray of shape (ndim,)
        R-hat statistic for each parameter.

    Examples
    --------
    >>> rhat = gelman_rubin('GWTC3_chain*.h5')
    >>> print(f"Max R-hat: {rhat.max():.3f}")
    >>> print(f"Converged: {np.all(rhat < 1.1)}")
    """
    files  = sorted(glob.glob(pattern))
    chains = [load_chain(f)[0] for f in files]

    m      = len(chains)
    n      = min(len(c) for c in chains)
    chains = np.array([c[:n] for c in chains])

    chain_means = chains.mean(axis=1)
    grand_mean  = chain_means.mean(axis=0)
    B           = n / (m - 1) * np.sum(
        (chain_means - grand_mean) ** 2, axis=0
    )
    W       = np.mean([np.var(c, axis=0, ddof=1) for c in chains], axis=0)
    var_hat = (1 - 1 / n) * W + B / n
    rhat    = np.sqrt(var_hat / W)

    return rhat


def convergence_report(pattern, param_names=None):
    """
    Load all chains and print a full convergence report including
    R-hat, ACL, and effective sample size.

    Parameters
    ----------
    pattern : str
        Glob pattern matching chain HDF5 files e.g. 'GWTC3_chain*.h5'.
    param_names : list of str, optional
        Parameter names for reporting. If None, loaded from first chain file.

    Examples
    --------
    >>> from vamana.utils import convergence_report
    >>> convergence_report('GWTC3_chain*.h5')
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching: {pattern}")

    # Load param_names from first chain if not provided
    if param_names is None:
        param_names = load_param_names(files[0])

    all_theta = np.concatenate([load_chain(f)[0] for f in files])

    print(f"{'='*50}")
    print(f"Convergence Report")
    print(f"{'='*50}")
    print(f"Chains:          {len(files)}")
    print(f"Samples/chain:   {len(load_chain(files[0])[0])}")
    print(f"Total samples:   {len(all_theta)}")
    print(f"{'='*50}")

    rhat = gelman_rubin(pattern)
    print(f"\nGelman-Rubin R-hat:")
    print(f"  Max R-hat:     {rhat.max():.3f}")
    print(f"  Mean R-hat:    {rhat.mean():.3f}")
    print(f"  Converged:     {np.all(rhat < 1.1)}")
    if param_names:
        worst_idx = np.argmax(rhat)
        print(f"  Worst param:   {param_names[worst_idx]} "
              f"(R-hat={rhat[worst_idx]:.3f})")

    print(f"\nAutocorrelation:")
    compute_acl(all_theta, param_names=param_names)
    print(f"{'='*50}")