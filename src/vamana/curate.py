import numpy as np
from vamana.conversions import z_to_dcovdz


def get_data_for_fitting(obsruns):
    """
    Organise raw observation data into analysis-ready arrays for faster
    likelihood calculation.

    For each observation run, PE samples from all events are concatenated
    into flat arrays. A 'breaks' list records the index boundaries between
    events, allowing per-event likelihood contributions to be computed
    efficiently. Analysis-independent quantities (comoving volume element,
    redshift factor, PE prior) are precomputed and combined into a single
    array to avoid redundant computation during sampling.

    The same precomputation is applied to injections, combining the
    comoving volume element, redshift factor, and injection prior PDF.

    Parameters
    ----------
    obsruns : dict
        Dictionary of ObsRun objects keyed by label, as stored in
        Data.obsruns. Each ObsRun must have:
        - events : dict of event dictionaries, each containing:
            - 'mass1_src' : np.ndarray, source-frame primary mass
            - 'mass2_src' : np.ndarray, source-frame secondary mass
            - 'spin1z' : np.ndarray, primary aligned spin component
            - 'spin2z' : np.ndarray, secondary aligned spin component
            - 'redshift' : np.ndarray, redshift
            - 'prior_pdf' : np.ndarray, PE prior probability density
        - injections : dict containing:
            - 'mass1_rec' : np.ndarray, recovered primary mass
            - 'mass2_rec' : np.ndarray, recovered secondary mass
            - 's1z_rec' : np.ndarray, recovered primary aligned spin
            - 's2z_rec' : np.ndarray, recovered secondary aligned spin
            - 'z_rec' : np.ndarray, recovered redshift
            - 'rec_pdf' : np.ndarray, injection prior PDF
            - 'w_rec' : np.ndarray, injection weights
            - 'analysis_time_yr' : float, total analysis time in years
            - 'ndraw' : float, total number of injections drawn

    Returns
    -------
    pe : dict
        Curated PE data keyed by obsrun label. Each value is a dict with:
        - 'parametric_data' : np.ndarray of shape (nsamples, 4),
            columns are [mass1_src, mass2_src, spin1z, spin2z]
        - 'breaks' : list of int, event boundary indices into parametric_data
        - 'log1pz' : np.ndarray, log(1 + redshift) for each sample
        - 'analysis_independent' : np.ndarray,
            z_to_dcovdz(z) / (1 + z) / prior_pdf for each sample

    injections : dict
        Curated injection data keyed by obsrun label. Each value is a dict with:
        - 'params_rec' : np.ndarray of shape (ninj, 4),
            columns are [mass1_rec, mass2_rec, s1z_rec, s2z_rec]
        - 'w_rec' : np.ndarray, injection weights
        - 'log1pz' : np.ndarray, log(1 + z_rec) for each injection
        - 'analysis_independent' : np.ndarray,
            z_to_dcovdz(z_rec) / (1 + z_rec) / rec_pdf for each injection
        - 'analysis_time_yr' : float, total analysis time in years
        - 'ndraw' : float, total number of injections drawn

    Notes
    -----
    Users who wish to include additional parameters (e.g. tilt angles)
    should define their own curation function with the same signature and
    pass it to Data(curate_fn=my_curate_fn).

    Examples
    --------
    >>> from vamana.curate import get_data_for_fitting
    >>> data = Data(curate_fn=get_data_for_fitting)
    >>> data.curate()
    >>> data.curated['O1O2'].pe['parametric_data'].shape
    (50000, 4)
    >>> data.curated['O1O2'].injections['params_rec'].shape
    (12345, 4)
    """
    pe = {}
    injections = {}

    for label, obsrun in obsruns.items():

        # --- PE data ---
        redshifts = []
        breakat, breaks = 0, [0]
        mass1, mass2 = [], []
        spin1z, spin2z = [], []
        pe_prior_pdf = []

        for event_name, event in obsrun.events.items():
            m1  = event['mass1_src']
            m2  = event['mass2_src']
            s1z = event['spin1z']
            s2z = event['spin2z']

            mass1  = np.append(mass1,  m1)
            mass2  = np.append(mass2,  m2)
            spin1z = np.append(spin1z, s1z)
            spin2z = np.append(spin2z, s2z)
            breakat += len(m1)
            breaks.append(breakat)
            redshifts    = np.append(redshifts, event['redshift'])
            pe_prior_pdf.extend(event['prior_pdf'])

        redshifts    = np.array(redshifts)
        pe_prior_pdf = np.array(pe_prior_pdf)

        pe[label] = {
            'parametric_data'      : np.transpose([mass1, mass2, spin1z, spin2z]),
            'breaks'               : breaks,
            'log1pz'               : np.log1p(redshifts),
            'analysis_independent' : z_to_dcovdz(redshifts) / (1 + redshifts) / pe_prior_pdf
        }

        # --- Injection data ---
        if obsrun.injections is not None:
            inj     = obsrun.injections
            m1_rec  = inj['mass1_rec']
            m2_rec  = inj['mass2_rec']
            s1z_rec = inj['s1z_rec']
            s2z_rec = inj['s2z_rec']
            z_rec   = inj['z_rec']

            injections[label] = {
                'params_rec'           : np.transpose([m1_rec, m2_rec, s1z_rec, s2z_rec]),
                'w_rec'                : inj['w_rec'],
                'log1pz'               : np.log1p(z_rec),
                'analysis_independent' : z_to_dcovdz(z_rec) / (1 + z_rec) / inj['rec_pdf'],
                'analysis_time_yr'     : inj['analysis_time_yr'],
                'ndraw'                : inj['ndraw']
            }

    return pe, injections