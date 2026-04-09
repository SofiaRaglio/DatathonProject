import mne
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import SlidingEstimator
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV, LinearRegression, LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


def get_default_pooling_window(
    epoch_type: str,
    response_pool_window: tuple[float, float] | None = None,
    feedback_pool_window: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    """Return default pooling window (seconds) for pooled decoding."""
    if epoch_type == "response":
        return response_pool_window if response_pool_window is not None else (-0.5, 0.0)
    if epoch_type == "feedback":
        return feedback_pool_window if feedback_pool_window is not None else (0.0, 1.0)
    return None


def apply_pooling_window(
    X: np.ndarray,
    times: np.ndarray,
    epoch_type: str,
    response_pool_window: tuple[float, float] | None = None,
    feedback_pool_window: tuple[float, float] | None = None,
) -> np.ndarray:
    """Restrict X to an event-specific pooling window before time averaging."""
    window = get_default_pooling_window(
        epoch_type,
        response_pool_window=response_pool_window,
        feedback_pool_window=feedback_pool_window,
    )
    if window is None:
        return X

    t_lo, t_hi = window
    mask = (times >= t_lo) & (times <= t_hi)
    if not np.any(mask):
        raise ValueError(
            f"No time points found in pooling window {window} for epoch '{epoch_type}'."
        )
    return X[:, :, mask]


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

def get_model(model_name=None, alphas=None, inner_cv_splits=5):
    """Return a regression model pipeline based on a name."""

    if alphas is None:
        alphas = np.logspace(-3, 3, 7)

    inner_cv = KFold(n_splits=inner_cv_splits, shuffle=True, random_state=0)

    # Default = RidgeCV
    if model_name is None or model_name == "ridge":
        model = RidgeCV(alphas=alphas, cv=inner_cv,
                        scoring="neg_mean_squared_error")

    elif model_name == "linear":
        model = LinearRegression()

    elif model_name == "svr":
        model = SVR(kernel="rbf")

    elif model_name == "rf":
        model = RandomForestRegressor(n_estimators=20,max_depth=8,max_features="sqrt",n_jobs=-1, random_state=0)

    elif model_name == "lasso":
        model = LassoCV(cv=inner_cv)

    elif model_name == "elasticnet":
        l1_ratios = [0.3, 0.5, 0.7]
        model = ElasticNetCV(cv=inner_cv, alphas=alphas, l1_ratio=l1_ratios, n_jobs=-1, max_iter=2500)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return make_pipeline(StandardScaler(), model)

def decoder_benchmark(X, y, runs, model_name=None, alphas=None, inner_cv_splits=5):
    """
    Modular decoder with pluggable regression models.
    """

    reg = get_model(model_name, alphas, inner_cv_splits)

    time_decoder = SlidingEstimator(reg, scoring=None)

    cv = LeaveOneGroupOut()
    splits = list(cv.split(X, y, groups=runs))

    n_times = X.shape[2]
    n_splits = len(splits)

    observed_corrs = np.zeros((n_splits, n_times))

    for i, (tr, te) in enumerate(splits):
        y_tr_eff, y_te_eff = y[tr], y[te]

        time_decoder.fit(X[tr], y_tr_eff)
        y_pred = time_decoder.predict(X[te])  # (n_te, n_times)

        observed_corrs[i, :] = corr_per_time(y_te_eff, y_pred)

    return observed_corrs.mean(axis=0)


def decoder_benchmark_pooled(X, y, runs, model_name=None, alphas=None, inner_cv_splits=5):
    """Decode using one pooled-in-time feature vector per trial.

    Time pooling is implemented as a mean across the time axis, resulting in
    features of shape (n_trials, n_channels).
    """

    reg = get_model(model_name, alphas, inner_cv_splits)

    # Pool across time so one model is fit per CV split (not one per time point).
    X_pooled = X.mean(axis=2)

    cv = LeaveOneGroupOut()
    splits = list(cv.split(X_pooled, y, groups=runs))

    observed_corrs = np.zeros(len(splits), dtype=float)

    for i, (tr, te) in enumerate(splits):
        y_tr_eff, y_te_eff = y[tr], y[te]

        reg.fit(X_pooled[tr], y_tr_eff)
        y_pred = reg.predict(X_pooled[te])

        corr_val, _ = pearsonr(y_te_eff, y_pred)
        observed_corrs[i] = corr_val

    # Keep output API consistent with the time-resolved path.
    return np.asarray([np.nanmean(observed_corrs)], dtype=float)


def run_decoding_for_quantity_and_epoch_benchmark(
    SUBJECT_INFO,
    DERIVATIVES_DIR,
    quantity,
    epoch_type,
    trial_type,
    sensor_type,
    baseline,
    LOWPASS,
    MODEL_NAME,
    pooled_decoding=False,
    response_pool_window: tuple[float, float] | None = None,
    feedback_pool_window: tuple[float, float] | None = None,
):
    
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

            event = epochs[epoch_type].copy().filter(None, LOWPASS).apply_baseline(baseline)
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
        X = concatenated_epochs_meg.get_data()
        
        runs = np.asarray(concatenated_metadata['run'])
        
        # Mask trials with missing targets (robust to non-float dtypes).
        mask = pd.notna(y)
        X = X[mask]
        y = y[mask]
        runs = runs[mask]
        
        if len(y) == 0:
            continue  # skip if nothing left after masking
        
        # Save the time axis from the last processed (non-empty) subject.
        times = concatenated_epochs.times


        if pooled_decoding:
            X = apply_pooling_window(
                X,
                concatenated_epochs.times,
                epoch_type,
                response_pool_window=response_pool_window,
                feedback_pool_window=feedback_pool_window,
            )
            corr = decoder_benchmark_pooled(
                X,
                y,
                runs,
                alphas=None,
                inner_cv_splits=5,
                model_name=MODEL_NAME,
            )
            # Pooled decoding has no real time axis; keep a placeholder for I/O consistency.
            times = np.asarray([np.nan], dtype=float)
        else:
            corr = decoder_benchmark(
                X,
                y,
                runs,
                alphas=None,
                inner_cv_splits=5,
                model_name=MODEL_NAME,
            )
            
        all_corrs.append(corr)
        subjects_included.append(subject)
    return {
        "all_corrs": all_corrs,
        "times": times,
        "subjects_included": subjects_included,
    }

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


def run_decoding_for_quantity_and_epoch(SUBJECT_INFO, DERIVATIVES_DIR, quantity, epoch_type, trial_type,sensor_type, baseline, LOWPASS):
    
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

            event = epochs[epoch_type].copy().filter(None, LOWPASS).apply_baseline(baseline)
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
        X = concatenated_epochs_meg.get_data()
        
        runs = np.asarray(concatenated_metadata['run'])
        
        # Mask trials with missing targets (robust to non-float dtypes).
        mask = pd.notna(y)
        X = X[mask]
        y = y[mask]
        runs = runs[mask]
        
        if len(y) == 0:
            continue  # skip if nothing left after masking
        
        # Save the time axis from the last processed (non-empty) subject.
        times = concatenated_epochs.times


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


def _extract_subject_ids_from_npz(data_npz, fallback_subject_ids=None):
    if fallback_subject_ids is not None:
        return [str(s) for s in fallback_subject_ids]

    ids = []
    for key in data_npz.files:
        if key.startswith("corr_r_sub"):
            ids.append(key.replace("corr_r_sub", ""))
    return sorted(ids)


def _pooled_subject_values(data_npz, subject_ids):
    vals = []
    kept_ids = []
    for sid in subject_ids:
        key = f"corr_r_sub{sid}"
        if key not in data_npz.files:
            continue
        arr = np.asarray(data_npz[key], dtype=float).ravel()
        if arr.size == 0:
            continue
        vals.append(arr[0])
        kept_ids.append(sid)
    vals = np.asarray(vals, dtype=float)
    finite_mask = np.isfinite(vals)
    vals = vals[finite_mask]
    kept_ids = [sid for sid, keep in zip(kept_ids, finite_mask) if keep]
    return vals, kept_ids


def _null_group_distribution_for_pooled(null_model_npz):
    if "arr_0" not in null_model_npz.files:
        return np.asarray([], dtype=float)

    null_arr = np.asarray(null_model_npz["arr_0"], dtype=float)
    if null_arr.ndim == 3:
        # Expected pooled shape: (subjects, permutations, 1)
        pooled_vals = null_arr[:, :, 0]
        valid_counts = np.sum(np.isfinite(pooled_vals), axis=0)
        summed = np.nansum(pooled_vals, axis=0)
        means = np.divide(
            summed,
            valid_counts,
            out=np.full(valid_counts.shape, np.nan, dtype=float),
            where=valid_counts > 0,
        )
        return means[np.isfinite(means)]
    if null_arr.ndim == 2:
        # Conservative fallback when a single pooled value per permutation is stored.
        if null_arr.shape[0] <= null_arr.shape[1]:
            valid_counts = np.sum(np.isfinite(null_arr), axis=0)
            summed = np.nansum(null_arr, axis=0)
            means = np.divide(
                summed,
                valid_counts,
                out=np.full(valid_counts.shape, np.nan, dtype=float),
                where=valid_counts > 0,
            )
        else:
            valid_counts = np.sum(np.isfinite(null_arr), axis=1)
            summed = np.nansum(null_arr, axis=1)
            means = np.divide(
                summed,
                valid_counts,
                out=np.full(valid_counts.shape, np.nan, dtype=float),
                where=valid_counts > 0,
            )
        return means[np.isfinite(means)]
    if null_arr.ndim == 1:
        return null_arr[np.isfinite(null_arr)]
    return np.asarray([], dtype=float)


def _resolve_window_for_plot(
    epoch_type,
    times_s,
    response_pool_window=None,
    feedback_pool_window=None,
):
    window = get_default_pooling_window(
        epoch_type,
        response_pool_window=response_pool_window,
        feedback_pool_window=feedback_pool_window,
    )
    if window is not None:
        return window

    t = np.asarray(times_s, dtype=float)
    finite_t = t[np.isfinite(t)]
    if finite_t.size:
        return float(np.min(finite_t)), float(np.max(finite_t))
    return None


def plot_pooled_decoding_interpretation(
    panels,
    null_model_npz,
    subject_ids=None,
    response_pool_window=None,
    feedback_pool_window=None,
    output_path=None,
):
    """Plot pooled decoding with explicit time-window and null-model interpretation.

    Parameters
    ----------
    panels:
        List of dicts with keys: ``label``, ``epoch_type``, ``data_npz``.
    null_model_npz:
        Loaded null model npz.
    subject_ids:
        Optional subject ordering to display.
    response_pool_window, feedback_pool_window:
        Optional custom pooling windows (seconds).
    output_path:
        Optional path to save the figure.
    """
    if len(panels) == 0:
        raise ValueError("panels must contain at least one entry")

    fig, axes = plt.subplots(
        len(panels),
        2,
        figsize=(12, 3.2 * len(panels)),
        gridspec_kw={"width_ratios": [1.2, 2.0]},
    )
    if len(panels) == 1:
        axes = np.asarray([axes])

    null_group_dist = _null_group_distribution_for_pooled(null_model_npz)

    for row_idx, panel in enumerate(panels):
        ax_window, ax_effect = axes[row_idx]

        label = panel["label"]
        epoch_type = panel["epoch_type"]
        data_npz = panel["data_npz"]

        if "times_s" in data_npz.files:
            times_s = np.asarray(data_npz["times_s"], dtype=float)
        else:
            times_s = np.asarray([np.nan], dtype=float)
        use_subject_ids = _extract_subject_ids_from_npz(data_npz, subject_ids)
        subject_vals, kept_ids = _pooled_subject_values(data_npz, use_subject_ids)

        obs_mean = float(np.mean(subject_vals)) if subject_vals.size else np.nan
        win = _resolve_window_for_plot(
            epoch_type,
            times_s,
            response_pool_window=response_pool_window,
            feedback_pool_window=feedback_pool_window,
        )

        # Left panel: physiological interpretation of the time pooling window.
        ax_window.axhline(0, color="0.65", lw=3, zorder=1)
        ax_window.axvline(0.0, color="0.35", lw=1.0, ls="--", zorder=2)
        if win is not None:
            ax_window.axvspan(win[0], win[1], color="tab:blue", alpha=0.25, zorder=0)
            ax_window.text(
                np.mean(win),
                0.1,
                f"pool: [{win[0]:.2f}, {win[1]:.2f}] s",
                ha="center",
                va="bottom",
                fontsize=9,
                color="tab:blue",
            )
        ax_window.set_ylim(-0.25, 0.35)
        ax_window.set_yticks([])
        ax_window.set_xlabel("Time around event (s)")
        ax_window.set_title(f"{label}: pooling window")
        if win is not None:
            pad = max(0.15, 0.2 * (win[1] - win[0]))
            ax_window.set_xlim(win[0] - pad, win[1] + pad)
        else:
            ax_window.set_xlim(-1.0, 1.0)

        # Right panel: observed pooled effect vs permutation null distribution.
        if null_group_dist.size:
            ax_effect.hist(
                null_group_dist,
                bins=25,
                density=True,
                color="0.75",
                edgecolor="white",
                alpha=0.9,
                label="Null (perm group mean)",
            )
            null_mu = float(np.nanmean(null_group_dist))
            null_sd = float(np.nanstd(null_group_dist, ddof=1))
            if np.isfinite(obs_mean):
                p_two_sided = (
                    1.0
                    + np.sum(np.abs(null_group_dist) >= np.abs(obs_mean))
                ) / (len(null_group_dist) + 1.0)
            else:
                p_two_sided = np.nan
            z_score = (obs_mean - null_mu) / null_sd if null_sd > 0 else np.nan
            stats_text = f"group mean r={obs_mean:.3f}\nnull mean={null_mu:.3f}\nz={z_score:.2f}, p={p_two_sided:.3f}"
            ax_effect.text(
                0.98,
                0.96,
                stats_text,
                transform=ax_effect.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75},
            )
        else:
            ax_effect.text(
                0.5,
                0.5,
                "No finite null values\n(check null-model run for this setting)",
                transform=ax_effect.transAxes,
                ha="center",
                va="center",
                fontsize=9,
            )

        if np.isfinite(obs_mean):
            ax_effect.axvline(obs_mean, color="tab:red", lw=2.0, label="Observed group mean")
        if subject_vals.size:
            ax_effect.scatter(
                subject_vals,
                np.full(subject_vals.shape, -0.03),
                transform=ax_effect.get_xaxis_transform(),
                marker="v",
                color="tab:blue",
                s=32,
                clip_on=False,
                label="Subject pooled r",
            )
            for x, sid in zip(subject_vals, kept_ids):
                ax_effect.text(
                    x,
                    -0.08,
                    sid,
                    transform=ax_effect.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="tab:blue",
                    clip_on=False,
                )

        ax_effect.axvline(0.0, color="k", lw=0.8, ls=":")
        ax_effect.set_title(f"{label}: pooled decoding vs null")
        ax_effect.set_xlabel("Pooled correlation (r)")
        ax_effect.set_yticks([])
        ax_effect.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig

