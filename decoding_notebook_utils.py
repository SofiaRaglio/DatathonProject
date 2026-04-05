import mne
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import SlidingEstimator
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
import pandas as pd


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

