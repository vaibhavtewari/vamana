import numpy as np
import h5py
import dynesty
from dynesty import pool as dypool
import vamana.priors as priors



def _save_chain(filepath, theta_samples, log_post, priors_spec, param_names, metadata):
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('samples/theta',    data=theta_samples)
        f.create_dataset('samples/log_post', data=log_post)

        # param_names in prior file order
        f.create_dataset('param_names',
                         data=np.array(param_names, dtype='S'))

        # Prior spec — one subgroup per parameter
        prior_grp = f.create_group('prior')
        for param, spec in priors_spec.items():
            pg = prior_grp.create_group(param)
            for k, v in spec.items():
                if v is None:
                    pg.attrs[k] = 'None'
                elif isinstance(v, list):
                    cleaned = [x if x is not None else float('nan') for x in v]
                    pg.create_dataset(k, data=np.array(cleaned, dtype=float))
                else:
                    pg.attrs[k] = v

        # Fixed parameters — saved separately for easy access
        fixed_grp = f.create_group('fixed_params')
        for param, spec in priors_spec.items():
            if spec['dist'] == 'fixed':
                fixed_grp.attrs[param] = spec['value']

        # Metadata
        meta_grp = f.create_group('metadata')
        for k, v in metadata.items():
            meta_grp.attrs[k] = v


def run_dynesty(analysis, nlive=500, dlogz=0.1, ncpu=1, sample='rslice', **kwargs):
    """
    Run the dynesty nested sampler.

    Proposal functions defined in the prior file are completely ignored
    by dynesty — it only calls prior_transform(u) which uses distribution
    functions only.

    Parameters
    ----------
    analysis : Analysis
    nlive    : int, optional. Default 500.
    dlogz    : float, optional. Default 0.1.
    ncpu     : int, optional. Default 1.
    sample   : str, optional. Default 'rslice'.
    **kwargs : passed to dynesty.NestedSampler.

    Returns
    -------
    dynesty.results.Results

    Examples
    --------
    >>> analysis.run(run_dynesty, nlive=1000, ncpu=20, sample='rwalk')
    """
    if ncpu > 1:
        with dypool.Pool(ncpu, analysis.log_likelihood,
                         analysis.prior_transform) as pool:
            sampler = dynesty.NestedSampler(
                pool.loglike, pool.prior_transform, analysis.ndim,
                nlive=nlive, sample=sample, pool=pool, **kwargs
            )
            sampler.run_nested(dlogz=dlogz)
    else:
        sampler = dynesty.NestedSampler(
            analysis.log_likelihood, analysis.prior_transform, analysis.ndim,
            nlive=nlive, sample=sample, **kwargs
        )
        sampler.run_nested(dlogz=dlogz)

    analysis.sampler = sampler
    return sampler.results


def _build_proposal_index(analysis, use_prior_draws=False):
    """
    Build indices for single-parameter proposals.

    Parameters with a proposal function defined in the prior file are
    added to proposal_index — these use their proposal function and
    require an MH correction.

    If use_prior_draws=True (run_mcmc_uniform_step):
        Parameters without a proposal function are added to prior_draw_index
        — these step in CDF space using <dist>_draw with delta_u.

    If use_prior_draws=False (run_mcmc):
        Parameters without a proposal function are left out of both indices
        — they go into am_idx and are handled by the AM covariance step.

    Returns
    -------
    proposal_index   : list of (j, draw_fnc, args, proposal_args, name)
    prior_draw_index : list of (j, draw_fnc, args, name)
    """
    proposal_index   = []
    prior_draw_index = []
    joint_indices    = set(j for dims in analysis.joint_proposals.values()
                           for j, _ in dims)

    for j, name in enumerate(analysis.param_names):
        if j in joint_indices:
            continue
        base_name = name if name in analysis.priors_spec \
                    else '_'.join(name.split('_')[:-1])
        spec      = analysis.priors_spec[base_name]

        if spec['dist'] == 'fixed':
            continue

        proposal = spec.get('proposal')
        args     = spec.get('args', [])

        if proposal is not None and not spec.get('is_joint', False):
            # Has an explicit proposal function — use it with MH correction
            draw_fnc      = getattr(priors, f"{proposal}_draw")
            proposal_args = spec.get('proposal_args', [])
            proposal_index.append((j, draw_fnc, args, proposal_args, name))
        elif use_prior_draws:
            # No proposal function — fall back to CDF step using <dist>_draw
            dist      = spec['dist']
            draw_name = f"{dist}_draw"
            if hasattr(priors, draw_name):
                draw_fnc = getattr(priors, draw_name)
                prior_draw_index.append((j, draw_fnc, args, name))

    return proposal_index, prior_draw_index


def _build_joint_proposal_index(analysis, dof_pool, delta_u=None):
    """
    Build index for joint proposals.

    Returns
    -------
    list of (proposal_name, indices, draw_fnc,
             dof_pool, proposal_args, member_args, delta_u)
    """
    index = []
    for proposal_name, dims in analysis.joint_proposals.items():
        first_base    = dims[0][1] if dims[0][1] in analysis.priors_spec \
                        else '_'.join(dims[0][1].split('_')[:-1])
        spec          = analysis.priors_spec[first_base]
        proposal_args = spec.get('proposal_args', [])

        member_args = {}
        seen        = set()
        for j, name in dims:
            base = name if name in analysis.priors_spec \
                   else '_'.join(name.split('_')[:-1])
            if base not in seen:
                member_args[base] = analysis.priors_spec[base].get('args', [])
                seen.add(base)

        draw_fnc = getattr(priors, f"{proposal_name}_draw")
        indices  = [j for j, _ in dims]

        index.append((proposal_name, indices, draw_fnc,
                       dof_pool, proposal_args, member_args, delta_u))
    return index


def _apply_proposals(theta_current, theta_proposed, proposal_index,
                     prior_draw_index, analysis, **kwargs):
    """
    Apply single-parameter proposals in theta space.
    Updates theta_proposed in-place. Returns accumulated log_mhc.

    Parameters with a proposal function (proposal_index) use their draw
    function and accumulate an MH correction.

    Parameters without a proposal function (prior_draw_index) step in
    CDF space using their distribution's own _draw function — log_mhc = 0
    for these, so they do not contribute to the MH correction.

    Each proposal draw function signature:
        draw(current, *args, **kwargs) -> (proposed, log_mhc)
    """
    log_mhc = 0.0

    # Parameters with explicit proposal functions — apply MH correction
    for j, draw_fnc, args, proposal_args, name in proposal_index:
        clean_args    = [x for x in args if x is not None]
        proposed_j, lmhc = draw_fnc(
            theta_current[j], *clean_args, *proposal_args,
            name=name, param_names=analysis.param_names,
            theta=theta_current, **kwargs
        )
        theta_proposed[j] = proposed_j
        log_mhc          += lmhc

    # Parameters without proposal functions — CDF step via <dist>_draw
    # log_mhc is ignored — CDF step is symmetric so prior draw always accepted.
    # log_mhc is still computed by the draw function in case it is ever
    # used as an explicit proposal function via the prior file.
    for j, draw_fnc, args, name in prior_draw_index:
        clean_args    = [x for x in args if x is not None]
        proposed_j, _ = draw_fnc(
            theta_current[j], *clean_args,
            name=name, param_names=analysis.param_names,
            theta=theta_proposed,
            theta_current=theta_current,
            **kwargs
        )
        theta_proposed[j] = proposed_j

    return log_mhc


def _apply_joint_proposals(theta_current, theta_proposed,
                            joint_proposal_index, analysis):
    """
    Apply joint proposals in theta space.
    Updates theta_proposed in-place. Returns accumulated log_mhc.

    Each joint draw function:
        draw(theta_group, dof_pool, **kwargs) -> (proposed_group, log_mhc)
    """
    log_mhc_joint = 0.0
    for proposal_name, indices, draw_fnc, dp, \
            proposal_args, member_args, delta_u_val in joint_proposal_index:
        theta_current_group = theta_current[indices]
        group_param_names   = [analysis.param_names[j] for j in indices]

        prp_group, lmhc = draw_fnc(
            theta_current_group, dp,
            proposal_args     = proposal_args,
            member_args       = member_args,
            delta_u           = delta_u_val,
            fixed_params      = analysis.fixed_params,
            param_names       = analysis.param_names,
            group_param_names = group_param_names
        )
        theta_proposed[indices] = prp_group
        log_mhc_joint          += lmhc

    return log_mhc_joint


def _theta_start(analysis):
    """
    Generate a random starting point in theta space by drawing from
    the prior transform at a random u point.
    """
    u = np.random.uniform(0.3, 0.7, analysis.ndim)
    return analysis.prior_transform(u)


def run_mcmc(analysis, nsteps=100000, nburn=10000, thin=1, chain_id=0,
             cov_update_interval=1000, cov_window=None,
             target_acceptance=0.234, output='mcmc_out',
             progress_interval=1000, theta_start=None):
    """
    Run a single adaptive Metropolis-Hastings MCMC chain in theta space.

    Works entirely in physical parameter space — no unit cube.
    AM covariance adapts parameters with no proposal function.
    Joint and single-parameter proposals are applied each step.
    Step size delta_u for proposals is computed automatically from
    nobs * ncomp via delta_u = sqrt(0.5 / (nobs * ncomp)).

    Parameters
    ----------
    analysis          : Analysis
    nsteps            : int, total steps including burn-in. Default 100000.
    nburn             : int, burn-in steps. Default 10000.
    thin              : int, thinning factor. Default 1.
    chain_id          : int, chain identifier. Default 0.
    cov_update_interval: int, steps between covariance updates. Default 1000.
    cov_window        : int or None, recent steps for covariance. Default None.
    target_acceptance : float, target acceptance rate. Default 0.234.
    output            : str, output file prefix. Default 'mcmc_out'.
    progress_interval : int, steps between progress prints. Default 1000.
    theta_start       : np.ndarray or None, starting point in theta space.

    Returns
    -------
    dict
    """
    ndim = analysis.ndim

    # Joint proposals — dof_pool scaled by nobs * ncomp
    min_nu   =     int(analysis.nobs * analysis.ncomp)
    max_nu   = 8 * int(analysis.nobs * analysis.ncomp)
    dof_pool = priors.get_dof(min_nu, max_nu, nsamp=10000)
    delta_u  = float(np.sqrt(0.5 / min_nu))

    proposal_index, prior_draw_index = _build_proposal_index(analysis, use_prior_draws=False)
    joint_proposal_index = _build_joint_proposal_index(
        analysis, dof_pool, delta_u=delta_u
    )

    # AM covariance for parameters with no proposal function — not joint, not single proposal
    joint_indices    = set(j for dims in analysis.joint_proposals.values()
                           for j, _ in dims)
    proposal_indices = set(j for j, _, _, _, _ in proposal_index)
    am_idx           = [j for j in range(ndim)
                        if j not in joint_indices
                        and j not in proposal_indices]
    n_am             = len(am_idx)

    # Initialise AM covariance for am_idx only
    global_scale  = 2.38 ** 2 / max(n_am, 1)
    prop_cov_sub  = np.eye(n_am) * global_scale * (0.01 ** 2)
    prop_chol_sub = np.linalg.cholesky(prop_cov_sub)
    chain         = np.zeros((nburn, n_am))

    # Initialise in theta space
    if theta_start is not None:
        if len(theta_start) != ndim:
            raise ValueError(
                f"theta_start has wrong shape: expected {ndim}, "
                f"got {len(theta_start)}. "
                f"Expected param order: {analysis.param_names}"
            )
        theta_current = theta_start.copy()
    else:
        theta_current = _theta_start(analysis)
    lp_current = analysis.log_posterior_theta(theta_current)

    n_post        = nsteps - nburn
    n_saved       = n_post // thin
    theta_samples = np.zeros((n_saved, ndim))
    log_post      = np.zeros(n_saved)

    n_accepted          = 0
    n_accepted_burn     = 0
    accepted_since_last = 0
    last_update_step    = 0
    save_idx            = 0
    rm_step_count       = 0

    print(f"Chain {chain_id} [AM theta]: nsteps={nsteps}, nburn={nburn}, "
          f"thin={thin}, ndim={ndim}, n_am={n_am}, "
          f"delta_u={delta_u:.4f}", flush=True)
    print(f"Expected saved samples: {n_saved}", flush=True)

    proposal_kwargs = dict(delta_u=delta_u, dof_pool=dof_pool)

    # --- Pre-sampling: find valid starting point ---
    lp_proposed = 1
    while lp_proposed > 0:
        step              = np.zeros(ndim)
        step[am_idx]      = prop_chol_sub @ np.random.randn(n_am)
        theta_proposed    = theta_current + step
        log_mhc           = _apply_proposals(
            theta_current, theta_proposed, proposal_index, prior_draw_index,
            analysis, **proposal_kwargs
        )
        log_mhc          += _apply_joint_proposals(
            theta_current, theta_proposed, joint_proposal_index, analysis
        )
        lp_proposed = analysis.log_posterior_theta(theta_proposed)

        if np.log(np.random.uniform()) < (lp_proposed - lp_current + log_mhc):
            theta_current = theta_proposed
            lp_current    = lp_proposed

    print(f"Chain {chain_id}: valid starting point found, "
          f"lp={lp_proposed:.3f}", flush=True)

    i             = 1
    theta_current = theta_proposed
    lp_current    = lp_proposed
    chain[0]      = theta_current[am_idx]

    # --- Main sampling loop ---
    while i < nsteps:

        # --- 2. Propose ---
        step              = np.zeros(ndim)
        step[am_idx]      = prop_chol_sub @ np.random.randn(n_am)
        theta_proposed    = theta_current + step
        log_mhc           = _apply_proposals(
            theta_current, theta_proposed, proposal_index, prior_draw_index,
            analysis, **proposal_kwargs
        )
        log_mhc          += _apply_joint_proposals(
            theta_current, theta_proposed, joint_proposal_index, analysis
        )
        lp_proposed = analysis.log_posterior_theta(theta_proposed)

        # --- 3. Accept/Reject ---
        if lp_proposed > 0:
            continue
        i += 1

        log_alpha = lp_proposed - lp_current + log_mhc
        if np.log(np.random.uniform()) < log_alpha:
            theta_current = theta_proposed
            lp_current    = lp_proposed
            n_accepted          += 1
            accepted_since_last += 1
            if i < nburn:
                n_accepted_burn += 1

        # --- 4. Store burn-in history (am_idx dims only) ---
        if i < nburn:
            chain[i] = theta_current[am_idx]

        # --- 5. Adapt AM covariance during burn-in ---
        if i < nburn and i % cov_update_interval == 0 and i > 2 * n_am:
            rm_step_count += 1

            steps_since_last    = i - last_update_step
            local_acc           = accepted_since_last / steps_since_last
            accepted_since_last = 0
            last_update_step    = i

            rm_step       = rm_step_count ** -0.6
            global_scale *= np.exp(rm_step * (local_acc - target_acceptance))

            window        = cov_window if cov_window else i
            recent        = chain[max(0, i - window): i]
            emp_cov_sub   = np.cov(recent.T)

            prop_cov_sub  = emp_cov_sub * global_scale + \
                            np.eye(n_am) * 1e-6

            try:
                prop_chol_sub = np.linalg.cholesky(prop_cov_sub)
            except np.linalg.LinAlgError:
                prop_cov_sub  = np.diag(np.diag(prop_cov_sub)) + \
                                np.eye(n_am) * 1e-6
                prop_chol_sub = np.linalg.cholesky(prop_cov_sub)

        # --- 6. Store thinned post burn-in samples ---
        if i >= nburn and (i - nburn) % thin == 0 and save_idx < n_saved:
            theta_samples[save_idx] = theta_current
            log_post[save_idx]      = lp_current
            save_idx               += 1

        # --- 7. Progress ---
        if i % progress_interval == 0:
            acceptance = n_accepted / i
            phase      = "burn-in" if i < nburn else "sampling"
            print(f"Chain {chain_id} | Step {i}/{nsteps} [{phase}] | "
                  f"acceptance: {acceptance:.3f} | "
                  f"log_post: {lp_current:.3f} | "
                  f"saved: {save_idx}/{n_saved}", flush=True)

    # Trim
    theta_samples = theta_samples[:save_idx]
    log_post      = log_post[:save_idx]

    n_accepted_post            = n_accepted - n_accepted_burn
    final_acceptance_rate      = n_accepted / nsteps
    final_acceptance_rate_burn = n_accepted_burn / nburn
    final_acceptance_rate_post = n_accepted_post / (nsteps - nburn)

    filepath = f"{output}_chain{chain_id}.h5"
    _save_chain(
        filepath, theta_samples, log_post,
        priors_spec = analysis.priors_spec,
        param_names = analysis.param_names,
        metadata    = {
            'chain_id'             : chain_id,
            'nsteps'               : nsteps,
            'nburn'                : nburn,
            'thin'                 : thin,
            'ndim'                 : ndim,
            'sampler'              : 'run_mcmc',
            'min_nu'               : min_nu,
            'max_nu'               : max_nu,
            'nobs'                 : analysis.nobs,
            'delta_u'              : delta_u,
            'acceptance_rate'      : final_acceptance_rate,
            'acceptance_rate_burn' : final_acceptance_rate_burn,
            'acceptance_rate_post' : final_acceptance_rate_post,
        }
    )

    print(f"\nChain {chain_id} done. "
          f"Acceptance (burn): {final_acceptance_rate_burn:.3f} | "
          f"Acceptance (post): {final_acceptance_rate_post:.3f} | "
          f"Saved: {save_idx} -> {filepath}", flush=True)

    return {
        'theta_samples'        : theta_samples,
        'log_post'             : log_post,
        'acceptance_rate'      : final_acceptance_rate,
        'acceptance_rate_burn' : final_acceptance_rate_burn,
        'acceptance_rate_post' : final_acceptance_rate_post,
        'proposal_cov'         : prop_cov_sub,
        'chain_id'             : chain_id,
        'thin'                 : thin,
        'nburn'                : nburn
    }


def run_mcmc_uniform_step(analysis, delta_u, nsteps=100000, nburn=10000,
                          thin=1, chain_id=0, target_acceptance=0.234,
                          output='mcmc_out', progress_interval=1000,
                          adapt_delta_u=True, delta_u_update_interval=1000,
                          theta_start=None):
    """
    Run a single MCMC chain in theta space using uniform step proposals.

    Fully equivalent to the original vamana MCMC implementation.
    Works entirely in physical parameter space — no unit cube.

    Single-parameter proposals and joint proposals are applied each step.
    Robbins-Monro adapts delta_u during burn-in to hit target acceptance.

    Parameters
    ----------
    analysis              : Analysis
    delta_u               : float, step size (max_dcdf equivalent)
    nsteps                : int, total steps including burn-in. Default 100000.
    nburn                 : int, burn-in steps. Default 10000.
    thin                  : int, thinning. Default 1.
    chain_id              : int, chain identifier. Default 0.
    target_acceptance     : float, target acceptance. Default 0.234.
    output                : str, output file prefix. Default 'mcmc_out'.
    progress_interval     : int, steps between prints. Default 1000.
    adapt_delta_u         : bool, adapt delta_u during burn-in. Default True.
    delta_u_update_interval: int, steps between delta_u updates. Default 1000.
    theta_start           : np.ndarray or None, starting point in theta space.

    Returns
    -------
    dict
    """
    ndim = analysis.ndim

    # Build proposal indices
    min_nu   = int(0.25 / delta_u ** 2)
    max_nu   = int(4.0 / delta_u ** 2)
    dof_pool = priors.get_dof(min_nu, max_nu, nsamp=10000)

    proposal_index, prior_draw_index = _build_proposal_index(analysis, use_prior_draws=True)
    joint_proposal_index = _build_joint_proposal_index(
        analysis, dof_pool, delta_u=delta_u
    )
    has_proposals = (len(proposal_index) > 0 or len(prior_draw_index) > 0
                     or len(joint_proposal_index) > 0)

    if not has_proposals:
        raise ValueError(
            "run_mcmc_uniform_step requires at least one proposal function "
            "defined in the prior file. Use run_mcmc instead."
        )

    proposal_kwargs = dict(delta_u=delta_u, dof_pool=dof_pool)

    # Initialise in theta space
    theta_current = theta_start.copy() if theta_start is not None \
                    else _theta_start(analysis)
    lp_current    = analysis.log_posterior_theta(theta_current)

    n_post        = nsteps - nburn
    n_saved       = n_post // thin
    theta_samples = np.zeros((n_saved, ndim))
    log_post      = np.zeros(n_saved)

    n_accepted             = 0
    n_accepted_burn        = 0
    accepted_since_last_du = 0
    last_update_step_du    = 0
    save_idx               = 0
    rm_step_count_du       = 0

    print(f"Chain {chain_id} [uniform step theta]: nsteps={nsteps}, "
          f"nburn={nburn}, thin={thin}, ndim={ndim}, "
          f"delta_u={delta_u:.4f}", flush=True)
    print(f"Expected saved samples: {n_saved}", flush=True)

    # --- Pre-sampling: find valid starting point ---
    lp_proposed = 1
    while lp_proposed > 0:
        theta_proposed = theta_current.copy()
        log_mhc        = _apply_proposals(
            theta_current, theta_proposed, proposal_index, prior_draw_index,
            analysis, **proposal_kwargs
        )
        log_mhc       += _apply_joint_proposals(
            theta_current, theta_proposed, joint_proposal_index, analysis
        )
        lp_proposed = analysis.log_posterior_theta(theta_proposed)

        if np.log(np.random.uniform()) < (lp_proposed - lp_current + log_mhc):
            theta_current = theta_proposed
            lp_current    = lp_proposed

    print(f"Chain {chain_id}: valid starting point found, "
          f"lp={lp_proposed:.3f}", flush=True)

    i             = 1
    theta_current = theta_proposed
    lp_current    = lp_proposed

    # --- Main sampling loop ---
    while i < nsteps:

        # --- 2. Propose in theta space ---
        theta_proposed = theta_current.copy()
        log_mhc        = _apply_proposals(
            theta_current, theta_proposed, proposal_index, prior_draw_index,
            analysis, **proposal_kwargs
        )
        log_mhc       += _apply_joint_proposals(
            theta_current, theta_proposed, joint_proposal_index, analysis
        )
        lp_proposed = analysis.log_posterior_theta(theta_proposed)

        # --- 3. Accept/Reject ---
        if lp_proposed > 0:
            continue
        i += 1

        log_alpha = lp_proposed - lp_current + log_mhc
        if np.log(np.random.uniform()) < log_alpha:
            theta_current = theta_proposed
            lp_current    = lp_proposed
            n_accepted            += 1
            accepted_since_last_du += 1
            if i < nburn:
                n_accepted_burn += 1

        # --- 4. Adapt delta_u during burn-in ---
        if adapt_delta_u and i < nburn and i % delta_u_update_interval == 0:
            rm_step_count_du += 1

            steps_since_last_du    = i - last_update_step_du
            local_acc_du           = accepted_since_last_du / steps_since_last_du
            accepted_since_last_du = 0
            last_update_step_du    = i

            rm_step_du  = rm_step_count_du ** -0.6
            delta_u    *= np.exp(rm_step_du * (local_acc_du - target_acceptance))
            delta_u     = float(np.clip(delta_u, 1e-6, 1.0))

            min_nu               = int(0.5 / delta_u ** 2)
            max_nu               = int(4.0 / delta_u ** 2)
            dof_pool             = priors.get_dof(min_nu, max_nu, nsamp=10000)
            joint_proposal_index = _build_joint_proposal_index(
                analysis, dof_pool, delta_u=delta_u
            )
            proposal_kwargs      = dict(delta_u=delta_u, dof_pool=dof_pool)

        # --- 5. Store thinned post burn-in samples ---
        if i >= nburn and (i - nburn) % thin == 0 and save_idx < n_saved:
            theta_samples[save_idx] = theta_current
            log_post[save_idx]      = lp_current
            save_idx               += 1

        # --- 6. Progress ---
        if i % progress_interval == 0:
            acceptance = n_accepted / i
            phase      = "burn-in" if i < nburn else "sampling"
            print(f"Chain {chain_id} | Step {i}/{nsteps} [{phase}] | "
                  f"acceptance: {acceptance:.3f} | "
                  f"log_post: {lp_current:.3f} | "
                  f"delta_u: {delta_u:.4f} | "
                  f"saved: {save_idx}/{n_saved}", flush=True)

    # Trim
    theta_samples = theta_samples[:save_idx]
    log_post      = log_post[:save_idx]

    n_accepted_post            = n_accepted - n_accepted_burn
    final_acceptance_rate      = n_accepted / nsteps
    final_acceptance_rate_burn = n_accepted_burn / nburn
    final_acceptance_rate_post = n_accepted_post / (nsteps - nburn)

    filepath = f"{output}_chain{chain_id}.h5"
    _save_chain(
        filepath, theta_samples, log_post,
        priors_spec = analysis.priors_spec,
        param_names = analysis.param_names,
        metadata    = {
            'chain_id'             : chain_id,
            'nsteps'               : nsteps,
            'nburn'                : nburn,
            'thin'                 : thin,
            'ndim'                 : ndim,
            'sampler'              : 'run_mcmc_uniform_step',
            'min_nu'               : min_nu,
            'max_nu'               : max_nu,
            'delta_u'              : delta_u,
            'acceptance_rate'      : final_acceptance_rate,
            'acceptance_rate_burn' : final_acceptance_rate_burn,
            'acceptance_rate_post' : final_acceptance_rate_post,
        }
    )

    print(f"\nChain {chain_id} done. "
          f"Acceptance (burn): {final_acceptance_rate_burn:.3f} | "
          f"Acceptance (post): {final_acceptance_rate_post:.3f} | "
          f"Saved: {save_idx} | delta_u: {delta_u:.6f} -> {filepath}",
          flush=True)

    return {
        'theta_samples'        : theta_samples,
        'log_post'             : log_post,
        'acceptance_rate'      : final_acceptance_rate,
        'acceptance_rate_burn' : final_acceptance_rate_burn,
        'acceptance_rate_post' : final_acceptance_rate_post,
        'delta_u'              : delta_u,
        'chain_id'             : chain_id,
        'thin'                 : thin,
        'nburn'                : nburn
    }