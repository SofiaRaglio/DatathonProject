import mne
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import SlidingEstimator
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
import pandas as pd
from dataclasses import dataclass
from typing import Optional


class TupleKeyEncoder(json.JSONEncoder):
    """JSON encoder that converts tuple dict keys to strings.

    Useful for serializing configs where PCA dicts use tuple keys
    like ``('real', 'imag')`` to group feature types.

    Works with both ``json.dumps`` and ``json.dump``.
    """

    def encode(self, o):
        return super().encode(self._convert_keys(o))

    def iterencode(self, o, _one_shot=False):
        return super().iterencode(self._convert_keys(o), _one_shot)

    def _convert_keys(self, obj):
        if isinstance(obj, dict):
            return {
                str(k) if isinstance(k, tuple) else k: self._convert_keys(v)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [self._convert_keys(i) for i in obj]
        return obj


@dataclass
class CPCAResult:
    """Results container for Circular PCA."""
    components: np.ndarray           # shape (n_components, n_features), complex
    scores: np.ndarray               # shape (n_samples, n_components), complex
    singular_values: np.ndarray      # shape (n_components,), real
    explained_variance: np.ndarray   # shape (n_components,), real
    explained_variance_ratio: np.ndarray  # shape (n_components,), real
    mean: np.ndarray                 # shape (n_features,), complex


class CircularPCA:
    """
    Circular (Proper) Complex PCA (CPCA).

    Operates on complex-valued data X ∈ C^{NxF} using the Hermitian
    covariance matrix C = (1/N) * X_c^H @ X_c, which guarantees real
    eigenvalues and orthonormal complex eigenvectors.

    The input is assumed to already be complex-valued (e.g. the analytic
    signal obtained via a prior Hilbert transform).

    Parameters
    ----------
    n_components : int or None
        Number of components to keep. If None, keeps all components.

    Raises
    ------
    ValueError
        If the input array is not complex-valued.
    """

    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.result_: Optional[CPCAResult] = None

    def fit_transform(self, X: np.ndarray) -> CPCAResult:
        """
        Fit CPCA to X and return a CPCAResult.

        Parameters
        ----------
        X : complex array of shape (n_samples, n_features)
            Must be complex-valued. For neuroimaging, this is typically
            the analytic signal obtained via scipy.signal.hilbert applied
            along the time axis before calling this method.

        Returns
        -------
        CPCAResult
        """
        X = np.asarray(X)

        if not np.iscomplexobj(X):
            raise ValueError(
                "Input X must be complex-valued. If you have real-valued data "
                "(e.g. fMRI/EEG timeseries), apply scipy.signal.hilbert(X, axis=0) "
                "first to obtain the analytic signal."
            )

        # --- Center the data ---
        mean = X.mean(axis=0)
        X_c = X - mean

        n_samples, n_features = X_c.shape
        n_components = self.n_components or min(n_samples, n_features)

        # --- SVD on complex data ---
        # X_c = U @ diag(S) @ V^H  →  C = X_c^H @ X_c / N = V @ diag(S²/N) @ V^H
        
        # Check for NaN values that would cause SVD to fail
        if np.any(np.isnan(X_c)):
            nan_count = np.sum(np.isnan(X_c))
            nan_ratio = nan_count / X_c.size
            raise ValueError(
                f"Input data contains NaN values ({nan_count} NaN values, {nan_ratio*100:.2f}% of data). "
                f"This typically occurs when MEG channels have bad data or the Hilbert transform fails. "
                f"Check your input data for disconnected sensors or artifacts."
            )
        
        # Check for Inf values that could cause numerical instability
        if np.any(np.isinf(X_c)):
            inf_count = np.sum(np.isinf(X_c))
            inf_ratio = inf_count / X_c.size
            raise ValueError(
                f"Input data contains Inf values ({inf_count} Inf values, {inf_ratio*100:.2f}% of data). "
                f"This typically occurs due to division by zero during standardization. "
                f"Check if any features have zero standard deviation."
            )
        
        # Perform SVD with error handling
        try:
            U, S, Vh = np.linalg.svd(X_c, full_matrices=False)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                f"SVD computation failed: {e}. "
                f"This may be due to numerical instability or ill-conditioned data. "
                f"Try reducing the number of PCA components or checking data quality."
            ) from e

        components = Vh[:n_components]                      # (k, n_features), complex
        scores = X_c @ Vh[:n_components].conj().T           # (n_samples, k), complex

        # --- Explained variance ---
        explained_variance = (S[:n_components] ** 2) / n_samples
        total_variance = (S ** 2).sum() / n_samples
        explained_variance_ratio = explained_variance / total_variance

        self.result_ = CPCAResult(
            components=components,
            scores=scores,
            singular_values=S[:n_components],
            explained_variance=explained_variance,
            explained_variance_ratio=explained_variance_ratio,
            mean=mean,
        )
        return self.result_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project new complex-valued data onto the fitted components.

        Parameters
        ----------
        X : complex array of shape (n_samples, n_features)

        Returns
        -------
        scores : complex array of shape (n_samples, n_components)
        """
        if self.result_ is None:
            raise RuntimeError("Call fit_transform before transform.")
        X = np.asarray(X)
        if not np.iscomplexobj(X):
            raise ValueError("Input X must be complex-valued.")
        X_c = X - self.result_.mean
        return X_c @ self.result_.components.conj().T


def _normalize_feature_types(feature_types, n_bands):
    """
    Normalize feature_types into a per-band list-of-lists format.

    Parameters
    ----------
    feature_types : str, list of str, or list of (str or list of str)
        Homogeneous: a single string or list of strings applied to all bands.
        Heterogeneous: a list of length n_bands where each element is a string
        or list of strings specifying features for that band.
    n_bands : int
        Number of frequency bands.

    Returns
    -------
    per_band : list of list of str
        Length n_bands, each element is a list of feature type strings.
    is_heterogeneous : bool
        True if per-band specification was detected.
    """
    VALID_FEATURES = {'amplitude', 'phase', 'real', 'imag'}

    if isinstance(feature_types, str):
        # Single string → homogeneous
        if feature_types not in VALID_FEATURES:
            raise ValueError(f"Unknown feature type: '{feature_types}'. Valid: {VALID_FEATURES}")
        return [[feature_types]] * n_bands, False

    if not isinstance(feature_types, list) or len(feature_types) == 0:
        raise ValueError("feature_types must be a non-empty string or list.")

    # Check if any element is a list → heterogeneous
    has_list = any(isinstance(ft, list) for ft in feature_types)

    if not has_list:
        # All strings → homogeneous, applied to every band
        for ft in feature_types:
            if ft not in VALID_FEATURES:
                raise ValueError(f"Unknown feature type: '{ft}'. Valid: {VALID_FEATURES}")
        return [list(feature_types)] * n_bands, False

    # Heterogeneous: must match n_bands
    if len(feature_types) != n_bands:
        raise ValueError(
            f"Heterogeneous feature_types length ({len(feature_types)}) "
            f"must match number of frequency bands ({n_bands})."
        )
    per_band = []
    for i, ft in enumerate(feature_types):
        if isinstance(ft, str):
            ft = [ft]
        for f in ft:
            if f not in VALID_FEATURES:
                raise ValueError(f"Unknown feature type '{f}' for band {i}. Valid: {VALID_FEATURES}")
        per_band.append(list(ft))
    return per_band, True


def _group_bands_by_features(per_band_features):
    """
    Group band indices by their feature type set.

    Parameters
    ----------
    per_band_features : list of list of str
        Per-band feature types (output of _normalize_feature_types).

    Returns
    -------
    groups : dict
        Keys are canonical group identifiers: a string if single feature,
        or a tuple of sorted strings if multiple features.
        Values are lists of band indices belonging to that group.
    """
    groups = {}
    for idx, features in enumerate(per_band_features):
        key = features[0] if len(features) == 1 else tuple(sorted(features))
        groups.setdefault(key, []).append(idx)
    return groups


def _validate_pca_option(pca, groups, is_heterogeneous):
    """
    Validate the pca option against feature groups.

    Parameters
    ----------
    pca : False, int, or dict
        PCA configuration.
    groups : dict
        Feature groups from _group_bands_by_features.
    is_heterogeneous : bool
        Whether feature types are heterogeneous.

    Returns
    -------
    pca_per_group : dict or None
        Maps group keys to int (n_components), or None if no PCA.
    """
    if pca is False or pca is None:
        return None

    if isinstance(pca, int):
        if is_heterogeneous:
            raise ValueError(
                "pca must be a dict (not int) when using heterogeneous feature types. "
                "Use a dict mapping feature group keys to n_components, e.g. "
                "{'amplitude': 10, ('real', 'imag'): 5}."
            )
        # Homogeneous: all groups get the same n_components
        return {key: pca for key in groups}

    if isinstance(pca, dict):
        # Normalize pca dict keys: sort tuple keys so order doesn't matter
        normalized_pca = {}
        for k, v in pca.items():
            norm_key = tuple(sorted(k)) if isinstance(k, tuple) else k
            if not isinstance(v, int) or v < 1:
                raise ValueError(f"pca[{k!r}] must be a positive integer, got {v!r}.")
            normalized_pca[norm_key] = v

        group_keys = set(groups.keys())
        pca_keys = set(normalized_pca.keys())
        missing = group_keys - pca_keys
        extra = pca_keys - group_keys
        if missing:
            raise ValueError(
                f"pca dict is missing entries for feature groups: {missing}. "
                f"Expected keys: {group_keys}"
            )
        if extra:
            raise ValueError(
                f"pca dict has extra keys not matching any feature group: {extra}. "
                f"Expected keys: {group_keys}"
            )
        return normalized_pca

    raise ValueError(f"pca must be False, an int, or a dict, got {type(pca).__name__}.")


def _apply_pca_to_group(data_band, n_components):
    """
    Apply standardization and CircularPCA to a complex data array.

    Parameters
    ----------
    data_band : np.ndarray
        Complex array of shape (n_epochs, n_channels_bands, n_times).
    n_components : int
        Number of PCA components to keep.

    Returns
    -------
    np.ndarray
        Complex array of shape (n_epochs, n_components, n_times).
    """
    n_epochs, n_channels_bands, n_times = data_band.shape
    X = data_band.transpose(1, 2, 0).reshape(n_channels_bands, n_times * n_epochs)

    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std[std == 0] = 1.0
    X = (X - mean) / std

    if np.any(np.isnan(X)):
        raise ValueError("Standardization produced NaN values. This may indicate constant features.")
    if np.any(np.isinf(X)):
        raise ValueError("Standardization produced Inf values. This may indicate numerical instability.")

    cpca = CircularPCA(n_components=n_components)
    result = cpca.fit_transform(X.T)
    return result.scores.reshape(n_times, n_epochs, n_components).transpose(1, 2, 0)


def _extract_features(data_band, feature_list):
    """
    Extract specified features from complex data.

    Parameters
    ----------
    data_band : np.ndarray
        Complex array of shape (n_epochs, n_channels, n_times).
    feature_list : list of str
        Features to extract: 'amplitude', 'phase', 'real', 'imag'.

    Returns
    -------
    np.ndarray
        Real array of shape (n_epochs, n_channels * len(feature_list), n_times).
    """
    parts = []
    for feature in feature_list:
        if feature == 'amplitude':
            parts.append(np.abs(data_band))
        elif feature == 'phase':
            parts.append(np.angle(data_band))
        elif feature == 'real':
            parts.append(np.real(data_band))
        elif feature == 'imag':
            parts.append(np.imag(data_band))
        else:
            raise ValueError(f"Unknown feature: {feature}")
    return np.concatenate(parts, axis=1)


def _get_time_freq_features(data, time_freq_options):
    """
    Extract time-frequency features from MEG epochs data.

    Processes the input data by filtering into frequency bands, applying Hilbert transform,
    and extracting specified features (amplitude, phase, real, or imaginary parts).
    Optionally applies PCA dimensionality reduction using CircularPCA.

    Supports both homogeneous features (same feature types for all bands) and
    heterogeneous features (different feature types per band).

    Parameters
    ----------
    data : mne.Epochs
        MEG epochs data to process. Should contain the raw sensor data.
    time_freq_options : dict
        Configuration dictionary with the following keys:
        - 'FREQ_BANDS' : list of tuples
            List of (low_freq, high_freq) tuples defining frequency bands to extract.
        - 'feature_types' : str, list of str, or list of (str or list of str)
            Homogeneous: a single string (e.g. 'amplitude') or list of strings
            (e.g. ['real', 'imag']) applied to all bands.
            Heterogeneous: a list of same length as FREQ_BANDS where each element
            is a string or list of strings specifying features for that band, e.g.
            [['amplitude'], ['real', 'imag'], ['real', 'imag']].
        - 'time_window' : tuple or None, optional
            (start_time, end_time) tuple to restrict analysis to specific time window.
        - 'pca' : int, dict, or False, optional
            Number of PCA components to keep.
            - False: no PCA.
            - int: same PCA for all bands (only valid with homogeneous features).
            - dict: per-feature-group PCA. Keys are group identifiers matching the
              feature type set (a string for single-feature groups, e.g. 'amplitude',
              or a tuple for multi-feature groups, e.g. ('real', 'imag')).
              Values are int (number of PCA components). Bands sharing the same
              feature types are grouped and PCA'd together.

    Returns
    -------
    features : np.ndarray
        Extracted features with shape (n_epochs, n_total_features, n_times).
    times : np.ndarray
        Time points corresponding to the features, shape (n_times,).

    Examples
    --------
    Homogeneous (backward compatible)::

        time_freq_options = {
            'FREQ_BANDS': [(8, 12), (13, 30)],
            'feature_types': ['amplitude'],
            'pca': 10,
        }

    Heterogeneous::

        time_freq_options = {
            'FREQ_BANDS': [(8, 12), (13, 30), (30, 60)],
            'feature_types': [['amplitude'], ['real', 'imag'], ['real', 'imag']],
            'pca': {'amplitude': 10, ('real', 'imag'): 5},
        }
    """
    freq_bands = time_freq_options['FREQ_BANDS']
    n_bands = len(freq_bands)

    per_band_features, is_heterogeneous = _normalize_feature_types(
        time_freq_options['feature_types'], n_bands
    )
    groups = _group_bands_by_features(per_band_features)
    pca_per_group = _validate_pca_option(
        time_freq_options.get('pca', False), groups, is_heterogeneous
    )

    # Filter per frequency band and apply Hilbert transform
    per_band_data = []
    times = None
    time_mask = None
    for l_freq, h_freq in freq_bands:
        _data_band = data.copy(
                ).filter(l_freq=l_freq, h_freq=h_freq, verbose=False
                ).apply_hilbert(verbose=False)
        data_band_array = _data_band.get_data()

        if time_freq_options.get('time_window', None) is not None:
            times = _data_band.times
            time_mask = (times >= time_freq_options['time_window'][0]) & (times <= time_freq_options['time_window'][1])
            data_band_array = data_band_array[:, :, time_mask]

        per_band_data.append(data_band_array)

    if times is None:
        times = _data_band.times
        time_mask = np.ones(len(times), dtype=bool)

    # Process each feature group: concatenate bands → optional PCA → extract features
    all_features = []
    for group_key, band_indices in groups.items():
        # Concatenate bands in this group along channel axis
        group_data = np.concatenate([per_band_data[i] for i in band_indices], axis=1)

        # Apply PCA if requested for this group
        if pca_per_group is not None and group_key in pca_per_group:
            group_data = _apply_pca_to_group(group_data, pca_per_group[group_key])

        # Determine feature list for this group
        feature_list = per_band_features[band_indices[0]]
        all_features.append(_extract_features(group_data, feature_list))

    # Concatenate all groups along channel axis
    return np.concatenate(all_features, axis=1), times[time_mask]


def load_epochs(subject,session, DERIVATIVES_DIR):
    """
    Load preprocessed (downsampled at 100Hz, [-1,1]s around task event) MEG epochs for one subject/session.

    Parameters
    ----------
    subject, session:
        Identifiers used in the derivatives folder structure.
    DERIVATIVES_DIR:
        Base derivatives directory containing `sub-*/ses-*/meg/`.

    Returns
    -------
    mne.Epochs | None
        Returns `None` if no epochs file matches the expected pattern.
    """
    base_path = Path(DERIVATIVES_DIR)/f'sub-{subject}/ses-{session}/meg'
    pattern = f"sub-{subject}_ses-{session}_task-decoding_desc-event_epo.fif"
    
    matches = list(base_path.glob(pattern))
    
    if matches:
        # If multiple matches, you can sort or choose the first one
        epochs_path = matches[0]
        print(f"Reading: {epochs_path}")
        epochs = mne.read_epochs(epochs_path)
    else:
        print(f"No matching epochs file for sub-{subject} ses-{session}")
        return None

    return epochs

def concatenate_behavior(subject, session, DERIVATIVES_DIR, SUBJECT_INFO):
    """
    Concatenate per-run behavioral TSV files into one DataFrame.

    Notes
    -----
    This function expects `SUBJECT_INFO[subject][session][-1]` to contain the
    number of runs for that subject/session.
    """
    behavioral_dfs = []
    base_path = Path(DERIVATIVES_DIR) / f'sub-{subject}' / f'ses-{session}' / 'beh'
    n_runs = SUBJECT_INFO[subject][session][-1]

    for run in range(1, n_runs + 1):
        run_path = (
            base_path
            / f'sub-{subject}_ses-{session}_task-ExplorePlus_run-{run:02d}_desc-formatted_beh.tsv'
        )
        df = pd.read_csv(run_path, sep='\t')
        df['run'] = run
        behavioral_dfs.append(df)

    behavioral_df = pd.concat(behavioral_dfs, ignore_index=True)
    return behavioral_df

def load_model_metadata(subject, session, DERIVATIVES_DIR, SUBJECT_INFO):
    """
    Build trial-level metadata for decoding.

    Combines:
    - an `IdealObserver_fittedVol.tsv` file (contains model-derived values)
    - behavioral TSVs (contains reward, RT, etc.)

    It also remaps `run` indices according to `SUBJECT_INFO[subject][session]`.
    """
    model_path = Path(DERIVATIVES_DIR) / f'sub-{subject}' / f'ses-{session}' / 'beh' / 'IdealObserver_fittedVol.tsv'
    io_df = pd.read_csv(model_path, sep='\t')
    io_df['rl'] = (io_df['arm_choice'] == "A").astype(int)
    
    beh_df = concatenate_behavior(subject,session, DERIVATIVES_DIR, SUBJECT_INFO)
    io_df['trial_idx']=beh_df.iloc[:,0]
    io_df['reward'] = beh_df['reward'].values
    io_df = io_df.dropna(subset=['reward'])
    beh_df = beh_df.dropna(subset=['reward'])
    io_df['run'] = beh_df['run'].values
    io_df['A_mean'] = beh_df['A_mean'].values
    io_df['B_mean'] = beh_df['B_mean'].values
    io_df['color_choice'] = beh_df['color_choice'].values
    io_df['color'] = (io_df['color_choice'] == "O").astype(int)
    io_df['forced'] = beh_df['forced'].values
    
    io_df['RT'] = beh_df['onset_response']-beh_df['onset_cue']
    # Keep only the runs we consider for this subject/session.
    runs_to_keep = SUBJECT_INFO[subject][session]
    filtered_df = io_df[io_df['run'].isin(runs_to_keep)]

    return filtered_df

def corr_per_time(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation at each time-point.

    Parameters
    ----------
    y_true:
        Shape (n_trials,).
    y_pred:
        Shape (n_trials, n_times). `y_pred[:, t]` is the prediction at time `t`.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_pred.ndim != 2:
        raise ValueError(f"y_pred must be 2D (n_trials, n_times), got shape {y_pred.shape}")
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1D (n_trials,), got shape {y_true.shape}")

    n_times_local = y_pred.shape[1]
    corrs = np.zeros(n_times_local)
    for t in range(n_times_local):
        corrs[t], _ = pearsonr(y_true, y_pred[:, t])
    return corrs

def decoder(X, y, runs, alphas=None, inner_cv_splits=5):
    """
    Run leave-one-run-out decoding using a ridge regressor and sliding window.

    Parameters
    ----------
    X:
        Array of shape (n_epochs, n_channels, n_times).
    y:
        Regression targets of shape (n_epochs,).
    runs:
        Group labels of shape (n_epochs,). Used by `LeaveOneGroupOut`.
    alphas:
        Candidate ridge `alpha` values for `RidgeCV`. If None, uses logspace(-3, 3).
    inner_cv_splits:
        Number of splits for the inner CV used by `RidgeCV`.

    Returns
    -------
    np.ndarray
        Mean Pearson r correlation over cross-validation splits for each time-point.
        Shape: (n_times,).
    """
    if alphas is None:
        # Default candidate range for ridge regularization.
        alphas = np.logspace(-3, 3, 7)

    inner_cv = KFold(n_splits=inner_cv_splits, shuffle=True, random_state=0)

    # RidgeCV will pick the best alpha on the training data each time .fit() is called
    reg = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=alphas, cv=inner_cv, scoring="neg_mean_squared_error")
    )

    time_decoder = SlidingEstimator(reg, scoring=None)

    cv = LeaveOneGroupOut()
    splits = list(cv.split(X, y, groups=runs))

    n_times = X.shape[2]
    n_splits = len(splits)


    observed_corrs = np.zeros((n_splits, n_times))

    for i, (tr, te) in enumerate(splits):
        y_tr_eff, y_te_eff = y[tr], y[te]

        time_decoder.fit(X[tr], y_tr_eff)
        y_pred = time_decoder.predict(X[te])       # (n_te, n_times)

        observed_corrs[i, :] = corr_per_time(y_te_eff, y_pred)

        
    observed_mean_corr = observed_corrs.mean(axis=0)  # (n_times,)
    return observed_mean_corr


def run_decoding_for_quantity_and_epoch(
    SUBJECT_INFO, DERIVATIVES_DIR, quantity, epoch_type, 
    trial_type,sensor_type, baseline, LOWPASS,
    time_freq_options=None):
    
    """
    High-level helper to run decoding for a given target `quantity`.

    It:
    1. Loads epochs per subject/session.
    2. Merges them with ideal observer + behavioral trial metadata.
    3. Filters trials by `trial_type` and selects sensor types + applies lowpass filter and baseline correction.
    4. Runs `decoder()` to produce a per-time correlation time-course.

    Returns
    -------
    dict
        Keys: ``all_corrs`` (list of arrays, shape ``(n_times,)`` per subject),
        ``times`` (1d array or ``None``), ``subjects_included`` (ids aligned with
        ``all_corrs``). Using a dict avoids tuple-unpack mistakes after edits or
        stale notebook imports.
    """
    all_corrs = []
    subjects_included = []
    times = None
    for subject, sessions in SUBJECT_INFO.items():
        subject_epochs = []
        subject_metadata = []
        # Index used to remap run numbers across sessions (legacy behavior).
        n = -1
        for session, runs in sessions.items():
            n=n+1
            io_df = load_model_metadata(subject, session, DERIVATIVES_DIR, SUBJECT_INFO)
            io_df['run'] = io_df['run']+n*8-1
            epochs = load_epochs(subject, session, DERIVATIVES_DIR)
            if epochs is None:
                continue
            if epoch_type not in epochs.event_id:
                continue           
            
            if time_freq_options is None:
                event = epochs[epoch_type].copy().filter(None, LOWPASS).apply_baseline(baseline)
            else:
                event = epochs[epoch_type].copy()
            subject_epochs.append(event)
            subject_metadata.append(io_df)

        if not subject_epochs:
            continue
        
        concatenated_epochs = mne.concatenate_epochs(subject_epochs, on_mismatch='ignore')
        concatenated_metadata = pd.concat(subject_metadata, ignore_index=True)
        
        if trial_type=='left':
            concatenated_epochs.metadata = concatenated_metadata
            concatenated_epochs = concatenated_epochs['arm_choice=="A"']
            concatenated_metadata=concatenated_metadata[concatenated_metadata['arm_choice']=="A"]
        elif trial_type=='right':
            concatenated_epochs.metadata = concatenated_metadata
            concatenated_epochs = concatenated_epochs['arm_choice=="B"']
            concatenated_metadata=concatenated_metadata[concatenated_metadata['arm_choice']=="B"]
        elif trial_type=='forced':
            concatenated_epochs.metadata = concatenated_metadata
            concatenated_epochs = concatenated_epochs['forced.notna()']
            concatenated_metadata=concatenated_metadata[concatenated_metadata['forced'].notna()]
        elif trial_type=='free':
            concatenated_epochs.metadata = concatenated_metadata
            concatenated_epochs = concatenated_epochs['forced.isna()']
            concatenated_metadata=concatenated_metadata[concatenated_metadata['forced'].isna()]
        elif trial_type=='repeat':
            concatenated_epochs.metadata = concatenated_metadata
            concatenated_epochs = concatenated_epochs['repeat==1']
            concatenated_metadata=concatenated_metadata[concatenated_metadata['repeat']==1]
        elif trial_type=='switch':
            concatenated_epochs.metadata = concatenated_metadata
            concatenated_epochs = concatenated_epochs['repeat==0']
            concatenated_metadata=concatenated_metadata[concatenated_metadata['repeat']==0]
        elif trial_type=='free_repeat':
            concatenated_epochs.metadata = concatenated_metadata
            concatenated_epochs = concatenated_epochs['forced.isna() and repeat==1'] 
            concatenated_metadata=concatenated_metadata.query('forced.isna() and repeat == 1')
        elif trial_type=='free_switch':
            concatenated_epochs.metadata = concatenated_metadata
            concatenated_epochs = concatenated_epochs['forced.isna() and repeat==0']
            concatenated_metadata=concatenated_metadata.query('forced.isna() and repeat == 0')
        else:
            concatenated_epochs.metadata = concatenated_metadata


        if quantity not in concatenated_metadata.columns:
            continue
        
        y = concatenated_metadata[quantity].to_numpy()
        if sensor_type == 'mag':
            concatenated_epochs_meg = concatenated_epochs.copy().pick_types(meg='mag')
        elif sensor_type == 'grad':
            concatenated_epochs_meg = concatenated_epochs.copy().pick_types(meg='grad')
        else:
            concatenated_epochs_meg = concatenated_epochs.copy().pick_types(meg=True)
        
        if time_freq_options is None:
            X = concatenated_epochs_meg.get_data()
            times = concatenated_epochs.times
        else:
            time_freq_options['time_freq_options'] = sensor_type
            X, times = _get_time_freq_features(concatenated_epochs, time_freq_options)

        
        runs = np.asarray(concatenated_metadata['run'])
        
        # Mask trials with missing targets (robust to non-float dtypes).
        mask = pd.notna(y)
        X = X[mask]
        y = y[mask]
        runs = runs[mask]
        
        if len(y) == 0:
            continue  # skip if nothing left after masking

        corr = decoder(
            X, y, runs,
            alphas=None,
            inner_cv_splits=5
        )
            
        all_corrs.append(corr)
        subjects_included.append(subject)
    return {
        "all_corrs": all_corrs,
        "times": times,
        "subjects_included": subjects_included,
    }
    