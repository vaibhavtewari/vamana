import os
import glob
import numpy as np
from vamana import read_data
from vamana import curate as curate_module


class ObsRun:
    """
    Represents a single gravitational wave observation run, containing
    the raw parameter estimation results and injections.

    Parameters
    ----------
    label : str
        A label identifying the observation run (e.g. 'O1O2', 'O3a').

    Attributes
    ----------
    label : str
        The observation run label.
    events : dict
        Dictionary of PE dictionaries keyed by event name.
    injections : dict or None
        Dictionary containing the injection data, or None if not yet added.

    Examples
    --------
    >>> obsrun = ObsRun('O1O2')
    >>> obsrun.label
    'O1O2'
    """

    def __init__(self, label):
        self.label = label
        self.events = {}
        self.injections = None

    def check(self, event_validator=None, injection_validator=None):
        """
        Validate the observation run data.

        Checks that events and injections have been added, and optionally
        runs user-supplied validator functions on each.

        Parameters
        ----------
        event_validator : callable, optional
            A function that takes a single event dictionary and raises an
            error if the data is invalid. Called for each event.
        injection_validator : callable, optional
            A function that takes the injections dictionary and raises an
            error if the data is invalid.

        Raises
        ------
        ValueError
            If no events or no injections have been added.

        Examples
        --------
        >>> def validate_event(pe):
        ...     required = ['mass1_src', 'mass2_src', 'spin1z', 'spin2z']
        ...     missing = [k for k in required if k not in pe]
        ...     if missing:
        ...         raise KeyError(f"Missing keys: {missing}")
        >>> obsrun.check(event_validator=validate_event)
        """
        if not self.events:
            raise ValueError(f"ObsRun '{self.label}' has no events")
        if self.injections is None:
            raise ValueError(f"ObsRun '{self.label}' has no injections")
        if event_validator:
            for name, event in self.events.items():
                event_validator(event)
        if injection_validator:
            injection_validator(self.injections)
        print(f"ObsRun '{self.label}': {len(self.events)} events, injections ok")


class CuratedObsRun:
    """
    Holds the curated (analysis-ready) data for a single observation run,
    produced by a curation function such as get_data_for_fitting.

    Parameters
    ----------
    label : str
        A label identifying the observation run (e.g. 'O1O2', 'O3a').

    Attributes
    ----------
    label : str
        The observation run label.
    pe : dict or None
        Curated parameter estimation data, ready for likelihood calculation.
    injections : dict or None
        Curated injection data, ready for likelihood calculation.

    Examples
    --------
    >>> curated = CuratedObsRun('O1O2')
    >>> curated.pe
    None
    >>> curated.injections
    None
    """

    def __init__(self, label):
        self.label = label
        self.pe = None
        self.injections = None


class Data:
    """
    Container for all gravitational wave observation data, including
    parameter estimates and injections across multiple observation runs.

    Manages loading, validation, and curation of data for use in the
    likelihood calculation.

    Parameters
    ----------
    curate_fn : callable, optional
        A function that takes the obsruns dictionary and returns
        (pe, injections) dictionaries keyed by label. Defaults to
        get_data_for_fitting from curate.py.

    Attributes
    ----------
    obsruns : dict
        Dictionary of ObsRun objects keyed by label.
    curated : dict
        Dictionary of CuratedObsRun objects keyed by label, populated
        after calling curate().

    Examples
    --------
    >>> from vamana.data import Data
    >>> data = Data()
    >>> data.add(pattern="/data/GWTC-1/*bbh*", reader="read_gwtc1", label="O1O2", nsamp=5000)
    >>> data.add_injections("/data/injections/o1o2.h5", reader="read_injections_gwtc1", label="O1O2", DETSNR_THR=8.0, NETSNR_THR=12.0)
    >>> data.check()
    >>> data.curate()
    >>> data.curated['O1O2'].pe
    >>> data.curated['O1O2'].injections
    """

    def __init__(self, curate_fn=None):
        self.obsruns = {}
        self._curate_fn = curate_fn if curate_fn is not None else curate_module.get_data_for_fitting
        self.curated = {}

    def add(self, pattern, reader, label, **kwargs):
        """
        Load parameter estimation files matching a glob pattern and add
        them to the specified observation run.

        The reader function is looked up by name in read_data.py. If the
        reader returns a nested dictionary (keyed by event name), each
        event is added individually. If it returns a single event
        dictionary, it is added directly.

        Parameters
        ----------
        pattern : str
            Glob pattern pointing to the PE files e.g. "/data/GWTC-1/*bbh*".
        reader : str
            Name of the reader function in read_data.py e.g. 'read_gwtc1'.
        label : str
            Label identifying the observation run e.g. 'O1O2'.
        **kwargs
            Additional keyword arguments passed to the reader function
            e.g. nsamp=5000.

        Raises
        ------
        FileNotFoundError
            If no files match the pattern.
        NotImplementedError
            If the reader function is not found in read_data.py.

        Examples
        --------
        >>> data.add(pattern="/data/GWTC-1/*bbh*", reader="read_gwtc1", label="O1O2", nsamp=5000)
        """
        if not hasattr(read_data, reader):
            raise NotImplementedError(f"No function '{reader}' found in read_data.py")
        if label not in self.obsruns:
            self.obsruns[label] = ObsRun(label)

        fn = getattr(read_data, reader)
        result = fn(pattern, **kwargs)

        # reader returned a nested dict of events e.g. read_gwtc1
        if isinstance(result, dict) and not any(isinstance(v, np.ndarray) for v in result.values()):
            for event_name, event_data in result.items():
                event_data['filename'] = event_name
                self.obsruns[label].events[event_name] = event_data
        # reader returned a single event dict
        else:
            filename = os.path.basename(pattern)
            result['filename'] = filename
            self.obsruns[label].events[filename] = result

    def add_injections(self, filename, reader, label, **kwargs):
        """
        Load an injections file and add it to the specified observation run.

        Parameters
        ----------
        filename : str
            Full path to the injections file.
        reader : str
            Name of the reader function in read_data.py e.g. 'read_injections_gwtc1'.
        label : str
            Label identifying the observation run e.g. 'O1O2'.
        **kwargs
            Additional keyword arguments passed to the reader function
            e.g. DETSNR_THR=8.0, NETSNR_THR=12.0.

        Raises
        ------
        FileNotFoundError
            If the injections file does not exist.
        NotImplementedError
            If the reader function is not found in read_data.py.

        Examples
        --------
        >>> data.add_injections("/data/injections/o1o2.h5", reader="read_injections_gwtc1", label="O1O2", DETSNR_THR=8.0, NETSNR_THR=12.0)
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        if not hasattr(read_data, reader):
            raise NotImplementedError(f"No function '{reader}' found in read_data.py")
        if label not in self.obsruns:
            self.obsruns[label] = ObsRun(label)

        fn = getattr(read_data, reader)
        inj = fn(filename, **kwargs)
        inj['filename'] = os.path.basename(filename)
        self.obsruns[label].injections = inj

    def check(self, event_validator=None, injection_validator=None):
        """
        Validate all observation runs.

        Calls check() on each ObsRun, optionally running user-supplied
        validator functions on events and injections.

        Parameters
        ----------
        event_validator : callable, optional
            A function that takes a single event dictionary and raises an
            error if the data is invalid.
        injection_validator : callable, optional
            A function that takes an injections dictionary and raises an
            error if the data is invalid.

        Raises
        ------
        ValueError
            If no observation runs have been added.

        Examples
        --------
        >>> data.check()
        ObsRun 'O1O2': 10 events, injections ok
        """
        if not self.obsruns:
            raise ValueError("No observation runs have been added")
        for label, obsrun in self.obsruns.items():
            obsrun.check(event_validator, injection_validator)

    def curate(self):
        """
        Run the curation function on all observation runs and store the
        results in self.curated.

        The curation function is called with the obsruns dictionary and
        should return (pe, injections) dictionaries keyed by label.
        Results are stored as CuratedObsRun objects in self.curated.

        Raises
        ------
        NotImplementedError
            If no curation function has been provided.

        Examples
        --------
        >>> data.curate()
        >>> data.curated['O1O2'].pe
        >>> data.curated['O1O2'].injections
        """
        if self._curate_fn is None:
            raise NotImplementedError("No curation function provided")
        pe, injections = self._curate_fn(self.obsruns)
        self.curated = {}
        for label in self.obsruns.keys():
            self.curated[label] = CuratedObsRun(label)
            self.curated[label].pe = pe.get(label)
            self.curated[label].injections = injections.get(label)