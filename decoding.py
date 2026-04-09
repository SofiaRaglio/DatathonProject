#!/usr/bin/env python
# coding: utf-8

# ## Benchmarking MEG decoding of decision drivers in learning under uncertainty
# 
# 

# Our goal is to identify neural correlates of key latent variables derived from Bayesian models of learning, such as expected reward, prediction error, and uncertainty. These variables are known to guide behavior in the task, but their neural representations remain incompletely characterized.
# We are particularly interested in:
# - Whether these latent variables are decodable from MEG signals
# - When they are represented in time relative to task events
# 

# In[1]:


# imports
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

# all the functions are in decoding_notebook_utils.py
from decoding_notebook_utils import (
    run_decoding_for_quantity_and_epoch_benchmark,
    plot_pooled_decoding_interpretation,
)


def _lowpass_tag_candidates(lowpass_value):
    tags = [
        str(lowpass_value),
        str(float(lowpass_value)),
        f"{float(lowpass_value):g}",
    ]
    # Keep insertion order while removing duplicates.
    return list(dict.fromkeys(tags))


def _load_decoding_npz(output_dir, quantity, epoch_type, trial_type, sensor_type, baseline, lowpass):
    tried_paths = []
    for lp_tag in _lowpass_tag_candidates(lowpass):
        path = output_dir / f"{quantity}_{epoch_type}_{trial_type}_{sensor_type}_{baseline}_{lp_tag}.npz"
        tried_paths.append(str(path))
        if path.exists():
            return np.load(path)

    tried = "\n".join(tried_paths)
    raise FileNotFoundError(
        "Could not find decoding output file. Tried:\n"
        f"{tried}\n"
        "If outputs do not exist yet, run with --run-decoding or generate them via submitit first."
    )


def _parse_cli_args():
    def _parse_pool_window(arg_value: str, arg_name: str):
        lower = arg_value.strip().lower()
        if lower in {"none", "null", "no"}:
            return None
        try:
            lo_raw, hi_raw = arg_value.split(",", maxsplit=1)
            return float(lo_raw), float(hi_raw)
        except ValueError as exc:
            raise ValueError(
                f"{arg_name} must be 'none' or two comma-separated floats, e.g. -0.5,0"
            ) from exc

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model-name",
        type=str,
        default="none",
        help="Model name used for output folder selection (none|ridge|linear|svr|rf)",
    )
    parser.add_argument(
        "--run-decoding",
        action="store_true",
        help="If set, run decoding before plotting. Default behavior is plot-only.",
    )
    parser.add_argument(
        "--pooled-decoding",
        "--pooling",
        dest="pooled_decoding",
        action="store_true",
        help="Use pooled decoding (single model over time-pooled features).",
    )
    parser.add_argument(
        "--response-pool-window",
        type=str,
        default="-0.5,0",
        help="Pooling window for response epochs in seconds as 'start,end'",
    )
    parser.add_argument(
        "--feedback-pool-window",
        type=str,
        default="0,1",
        help="Pooling window for feedback epochs in seconds as 'start,end'",
    )
    args, _unknown = parser.parse_known_args()
    model_name = None if args.model_name.lower() in {"none", "null"} else args.model_name
    response_pool_window = _parse_pool_window(args.response_pool_window, "--response-pool-window")
    feedback_pool_window = _parse_pool_window(args.feedback_pool_window, "--feedback-pool-window")
    return (
        model_name,
        args.run_decoding,
        args.pooled_decoding,
        response_pool_window,
        feedback_pool_window,
    )


# The data we shared are from 3 subjects: subject 1, 11 and 19. 
# 
# The data are in the folder "derivatives" and in each subject folder there are two subfolders corresponding to the two sessions. In our current analysis we concatenate the data from the two sessions. In each session foldere there are two subfolders: "meg" and "beh". 
# 
# In "meg" you will find the MEG data, already epoched around task events (timewindow=[-1,1]s around the event) aggregated for all the 8 recorded runs. The number of trials might differ across subjects and sessions because some trials have missed responses. 
# 
# In "beh" you will find 8 separate tsv files corresponding to the behavior in the 8 runs. You will also find a tsv file named "IdealObserver_fittedVol" which contains the labels for the decoding, obtained from a Bayesian ideal observer performing the task, with a volatility fitted to the subject's choice. 

# In[2]:

(
    MODEL_NAME,
    RUN_DECODING,
    POOLED_DECODING,
    RESPONSE_POOL_WINDOW,
    FEEDBACK_POOL_WINDOW,
) = _parse_cli_args()
DECODE_MODE = "pooled" if POOLED_DECODING else "time_resolved"
# paths to the data and the output directory

# DERIVATIVES_DIR = Path("/where/you/downloaded/data")
# OUTPUT_DIR = Path("where/you/want/to/save/outputs")
DERIVATIVES_DIR = Path("/data/parietal/store3/work/ggomezji/projects/DatathonProject/DatathonProject/derivatives")
OUTPUT_DIR = DERIVATIVES_DIR.parent / "decoding_outputs" / f"model_{MODEL_NAME if MODEL_NAME is not None else 'RidgeCV'}"
OUTPUT_DIR = OUTPUT_DIR / DECODE_MODE
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# In[3]:


# subjects and sessions to run the decoding analysis

SUBJECT_INFO = {
    '01': {
        '4': [1,2,3,4,5,6,7,8],  
        '5': [1,2,3,4,5,6,7,8],  
    },
    '11': {
        '4': [1,2,3,4,5,6,7,8],        
        '5': [1,2,3,4,5,6,7,8]           
    },
    '19': {
        '2': [1,2,3,4,5,6,7,8],        
        '3': [1,2,3,4,5,6,7,8]           
    },
}


# The task events (epoch_type) are "cue", "response", "feedback", "questions", "answers". We will only be interestend in "response" and "feedback". 
# 
# The ideal observer quantities (quantity) that can be predicted from the neural activity are many, but we will be intereseted in 3: "ER_diffRS_z", "EU_diffRS_z" and "PE_z".
# 
# The trial types are several: "free", "forced", "repeat", "switch", "left", "right". We will only be interested in "free".
# 
# The possible sensor types are "mag", "grad", "meg" (which corresponds to both "mag" and "grad"). For our attempts so far it seems that we obtain better decoding with "mag".
# 
# Feel free to play with baseline and lowpass filtering. 

# baseline info: In MNE, baseline is always `baseline = (tmin, tmax)`
# some examples
# 
# Classic pre-stimulus: `baseline = (-0.2, 0)` 
# - Uses 200 ms before stimulus onset
# - Centers each trial around its pre-event activity
# - When to use it: Epochs are stimulus-locked, want stimulus-evoked responses and no important signal exists before 0
# 
# Longer pre-stimulus: `baseline = (-0.5, 0)`
# - More stable estimate (averages more time)
# - When to use it: Noisy data and slow fluctuations present
# 
# Very short baselines: `baseline = (-0.1, 0)`
# - Minimal correction, very local
# - When to use it: Rapid designs and you want to preserve slow dynamics
# 
# Far before the event: `baseline = (-1.0, -0.5)`
# - Uses an earlier time window, avoiding immediate pre-stimulus
# - When to use it: You suspect anticipation effects and pre-stimulus buildup

# In[4]:


# decoding parameters
trial_type = 'free'
sensor_type = 'mag'
baseline = None      
LOWPASS = 15
LOWPASS_SAVE_TAG = str(float(LOWPASS))
# Batch: all three quantities × event windows
DECODING_JOBS = [
    ("ER_diffRS_z", "response"),
    ("EU_diffRS_z", "response"),
    ("PE_z", "feedback"),
]



# The baseline decoder we propose for the benchmarking is a ridge regression with crossvalidated alpha using leave-one-run-out crossvalidation.
# 
# We use mne SlidingEstimator(), which means that we have one decoder for each time point in our epoch.
# 
# If you run the cell below you will compute the decoding results (pearson correlation between real and predicted quantity at each time point): you can skip it and load the decoding results from the output folder with the cell below.

# In[ ]:


# Optional: run the decoding. One call processes every subject in SUBJECT_INFO.
if RUN_DECODING:
    last_decoding = None
    for q, ev in DECODING_JOBS:
        key = f"{q}_{ev}"
        print(f"=== {key} ===")
        out = run_decoding_for_quantity_and_epoch_benchmark(
            SUBJECT_INFO,
            DERIVATIVES_DIR,
            q,
            ev,
            trial_type,
            sensor_type,
            baseline,
            LOWPASS,
            MODEL_NAME=MODEL_NAME,
            pooled_decoding=POOLED_DECODING,
            response_pool_window=RESPONSE_POOL_WINDOW,
            feedback_pool_window=FEEDBACK_POOL_WINDOW,
        )
        last_decoding = out
        all_corrs = out["all_corrs"]
        times = out["times"]
        subjects_included = out["subjects_included"]

        # Per-job .npz: one correlation vector per subject that was included
        save_kw = {}
        if times is not None:
            save_kw["times_s"] = np.asarray(times, dtype=float)
        for sub_id, corr in zip(subjects_included, all_corrs):
            save_kw[f"corr_r_sub{sub_id}"] = np.asarray(corr, dtype=float)
        out_npz = OUTPUT_DIR / f"{key}_{trial_type}_{sensor_type}_{baseline}_{LOWPASS_SAVE_TAG}.npz"
        if save_kw:
            np.savez_compressed(out_npz, **save_kw)
            print(f"Saved {out_npz} (n_subjects={len(all_corrs)})")
        else:
            print(f"No arrays to save for {key} (skipped .npz)")


# Instead of computing the decoding we can load the already computed values with the current model. 
# 
# We can also load a previously computed null-model. The null-model is created with the exact same decoding procedure, but shuffling the training set with 200 permutations. The loaded null_model_correlation.npz will have shape (subjects,permutations,time-points).

# In[10]:


# Load decoding results for ER, EU, and PE (same filenames as saved by the decoding cell)

ER = _load_decoding_npz(
    OUTPUT_DIR,
    "ER_diffRS_z",
    "response",
    trial_type,
    sensor_type,
    baseline,
    LOWPASS,
)
EU = _load_decoding_npz(
    OUTPUT_DIR,
    "EU_diffRS_z",
    "response",
    trial_type,
    sensor_type,
    baseline,
    LOWPASS,
)
PE = _load_decoding_npz(
    OUTPUT_DIR,
    "PE_z",
    "feedback",
    trial_type,
    sensor_type,
    baseline,
    LOWPASS,
)

#load the null model
null_model = np.load(OUTPUT_DIR / "null_model_correlation.npz")


# In[11]:

SUBJECT_IDS = sorted(SUBJECT_INFO.keys())
COLORS = {"01": "C0", "11": "C1", "19": "C2"}


def _plot_quantity(ax, data_npz, title):
    t = np.asarray(data_npz["times_s"], dtype=float)
    for sid in SUBJECT_IDS:
        key = f"corr_r_sub{sid}"
        if key not in data_npz.files:
            continue
        ax.plot(
            t,
            np.asarray(data_npz[key], dtype=float),
            label=f"sub-{sid}",
            color=COLORS.get(sid, None),
            alpha=0.9,
        )
    ax.plot(t, np.mean(np.mean(null_model['arr_0'], axis=1),axis=0), label="null", color='k', alpha=0.85)
    ax.axhline(0.0, color="k", lw=0.5, ls=":")
    ax.axvline(0.0, color="gray", lw=0.6, ls="--")
    ax.set_ylabel("Correlation (r)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)


if not POOLED_DECODING:
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    _plot_quantity(axes[0], ER, "ER_diffRS_z (response)")
    _plot_quantity(axes[1], EU, "EU_diffRS_z (response)")
    _plot_quantity(axes[2], PE, "PE_z (feedback)")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "decoding_results.png", dpi=300)
    # plt.show()
else:
    def _pooled_subject_table(data_npz, title):
        rows = []
        for sid in SUBJECT_IDS:
            key = f"corr_r_sub{sid}"
            val = float(data_npz[key][0]) if key in data_npz.files else float("nan")
            rows.append({"subject": f"sub-{sid}", "pooled_r": val})
        df = pd.DataFrame(rows)
        safe_title = title.replace(" ", "_").replace("(", "").replace(")", "")
        df.to_csv(OUTPUT_DIR / f"{safe_title}_pooled_table.csv", index=False)
        print(f"\n{title} (pooled)")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    _pooled_subject_table(ER, "ER_diffRS_z (response)")
    _pooled_subject_table(EU, "EU_diffRS_z (response)")
    _pooled_subject_table(PE, "PE_z (feedback)")

    pooled_panels = [
        {"label": "ER_diffRS_z", "epoch_type": "response", "data_npz": ER},
        {"label": "EU_diffRS_z", "epoch_type": "response", "data_npz": EU},
        {"label": "PE_z", "epoch_type": "feedback", "data_npz": PE},
    ]
    plot_pooled_decoding_interpretation(
        pooled_panels,
        null_model,
        subject_ids=SUBJECT_IDS,
        response_pool_window=RESPONSE_POOL_WINDOW,
        feedback_pool_window=FEEDBACK_POOL_WINDOW,
        output_path=OUTPUT_DIR / "decoding_results_pooled_interpretation.png",
    )


# For ER and EU (decoded at the response), we use two scalars per subject: mean decoding correlation in the 200 ms before the response, and in the 200 ms after the response.
# 
# For PE (decoded at feedback), we use a single scalar per subject: mean correlation between 250 and 500 ms after the feedback event.

# In[12]:


# Benchmarking scalars (times in decoding outputs are seconds):
# — ER & EU (response-locked): mean r in [-200 ms, 0) pre-response and in [0, 200 ms] post-response
# — PE (feedback-locked): mean r in [250, 500] ms after feedback only
SUBJECT_IDS = sorted(SUBJECT_INFO.keys())


def _mean_corr_interval(times_s, corr, t_lo, t_hi, *, lo_open: bool, hi_open: bool):
    t = np.asarray(times_s, dtype=float)
    r = np.asarray(corr, dtype=float)
    mask_lo = t > t_lo if lo_open else t >= t_lo
    mask_hi = t < t_hi if hi_open else t <= t_hi
    mask = mask_lo & mask_hi
    if not np.any(mask):
        return float("nan")
    return float(np.nanmean(r[mask]))


def table_response_locked(data_npz):
    t = data_npz["times_s"]
    rows = []
    for sid in SUBJECT_IDS:
        key = f"corr_r_sub{sid}"
        if key not in data_npz.files:
            rows.append(
                {
                    "subject": f"sub-{sid}",
                    "mean_r_pre_response_200ms": np.nan,
                    "mean_r_post_response_200ms": np.nan,
                }
            )
            continue
        r = data_npz[key]
        rows.append(
            {
                "subject": f"sub-{sid}",
                "mean_r_pre_response_200ms": _mean_corr_interval(
                    t, r, -0.2, 0.0, lo_open=False, hi_open=True
                ),
                "mean_r_post_response_200ms": _mean_corr_interval(
                    t, r, 0.0, 0.2, lo_open=False, hi_open=False
                ),
            }
        )
    return pd.DataFrame(rows)


def table_feedback_pe(data_npz):
    t = data_npz["times_s"]
    rows = []
    for sid in SUBJECT_IDS:
        key = f"corr_r_sub{sid}"
        if key not in data_npz.files:
            rows.append({"subject": f"sub-{sid}", "mean_r_post_feedback_250_500ms": np.nan})
            continue
        r = data_npz[key]
        rows.append(
            {
                "subject": f"sub-{sid}",
                "mean_r_post_feedback_250_500ms": _mean_corr_interval(
                    t, r, 0.25, 0.5, lo_open=False, hi_open=False
                ),
            }
        )
    return pd.DataFrame(rows)


def table_timecourse(data_npz, null_npz):
    """Export full time-course curves per subject with null mean overlay."""
    t = np.asarray(data_npz["times_s"], dtype=float)
    out = pd.DataFrame({"time_s": t})

    for sid in SUBJECT_IDS:
        key = f"corr_r_sub{sid}"
        if key in data_npz.files:
            out[f"sub-{sid}"] = np.asarray(data_npz[key], dtype=float)
        else:
            out[f"sub-{sid}"] = np.nan

    if "arr_0" in null_npz.files:
        null_arr = np.asarray(null_npz["arr_0"], dtype=float)
        # null_arr expected shape: (subjects, permutations, times)
        if null_arr.ndim == 3 and null_arr.shape[-1] == len(t):
            out["null_mean"] = np.nanmean(null_arr, axis=(0, 1))
    return out


if not POOLED_DECODING:
    for title, npz, table_fn in [
        ("ER_diffRS_z (response)", ER, table_response_locked),
        ("EU_diffRS_z (response)", EU, table_response_locked),
        ("PE_z (feedback)", PE, table_feedback_pe),
    ]:
        df = table_fn(npz)
        df.to_csv(OUTPUT_DIR / f"{title}_table.csv", index=False)
        tc_df = table_timecourse(npz, null_model)
        tc_df.to_csv(OUTPUT_DIR / f"{title}_timecourse_table.csv", index=False)
        print(f"\n{title}")
        print(table_fn(npz).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# In[ ]:




