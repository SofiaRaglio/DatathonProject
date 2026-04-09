#!/usr/bin/env python
"""Submit decoding jobs to a SLURM cluster with submitit.

Each submitted job runs one (quantity, epoch_type) decoding configuration and
saves a per-job .npz output, matching the notebook naming convention.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import submitit
from mne.decoding import SlidingEstimator
from sklearn.model_selection import LeaveOneGroupOut

from decoding_notebook_utils import (
    apply_pooling_window,
    corr_per_time,
    get_model,
    load_epochs,
    load_model_metadata,
    run_decoding_for_quantity_and_epoch_benchmark,
)


DEFAULT_SUBJECT_INFO = {
    "01": {"4": [1, 2, 3, 4, 5, 6, 7, 8], "5": [1, 2, 3, 4, 5, 6, 7, 8]},
    "11": {"4": [1, 2, 3, 4, 5, 6, 7, 8], "5": [1, 2, 3, 4, 5, 6, 7, 8]},
    "19": {"2": [1, 2, 3, 4, 5, 6, 7, 8], "3": [1, 2, 3, 4, 5, 6, 7, 8]},
}

DEFAULT_DECODING_JOBS = [
    ("ER_diffRS_z", "response"),
    ("EU_diffRS_z", "response"),
    ("PE_z", "feedback"),
]

DEFAULT_NULL_REFERENCE_JOB = ("ER_diffRS_z", "response")
DEFAULT_NULL_PERMUTATIONS = 200
DEFAULT_NULL_RANDOM_SEED = 0

CLUSTER_SUBMITIT_CONFIG = {
    "slurm_partition": "normal,normal-best,parietal",
    "tasks_per_node": 1,
    "slurm_cpus_per_task": 20,
    "timeout_min": 900,
}


def _parse_baseline(arg_value: str) -> tuple[float, float] | None:
    lower = arg_value.strip().lower()
    if lower in {"none", "null", "no"}:
        return None
    try:
        lo_raw, hi_raw = arg_value.split(",", maxsplit=1)
        return float(lo_raw), float(hi_raw)
    except ValueError as exc:
        raise ValueError(
            "--baseline must be 'none' or two comma-separated floats, e.g. -0.2,0"
        ) from exc


def _parse_jobs(job_args: Iterable[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for item in job_args:
        try:
            quantity, epoch_type = item.split(":", maxsplit=1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid job '{item}'. Expected format QUANTITY:EPOCH_TYPE"
            ) from exc
        parsed.append((quantity, epoch_type))
    return parsed


def _parse_pool_window(arg_value: str, arg_name: str) -> tuple[float, float] | None:
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


def _save_job_output(
    output_dir: Path,
    quantity: str,
    epoch_type: str,
    trial_type: str,
    sensor_type: str,
    baseline: tuple[float, float] | None,
    lowpass: float,
    out: dict,
) -> Path:
    all_corrs = out["all_corrs"]
    times = out["times"]
    subjects_included = out["subjects_included"]

    baseline_tag = "None" if baseline is None else f"({baseline[0]},{baseline[1]})"
    out_npz = (
        output_dir
        / f"{quantity}_{epoch_type}_{trial_type}_{sensor_type}_{baseline_tag}_{lowpass}.npz"
    )

    save_kw: dict[str, np.ndarray] = {}
    if times is not None:
        save_kw["times_s"] = np.asarray(times, dtype=float)
    for sub_id, corr in zip(subjects_included, all_corrs):
        save_kw[f"corr_r_sub{sub_id}"] = np.asarray(corr, dtype=float)

    if save_kw:
        np.savez_compressed(out_npz, **save_kw)
    return out_npz


def run_single_decoding_job(
    *,
    derivatives_dir: str,
    output_dir: str,
    quantity: str,
    epoch_type: str,
    trial_type: str,
    sensor_type: str,
    baseline: tuple[float, float] | None,
    lowpass: float,
    model_name: str | None,
    pooled_decoding: bool = False,
    response_pool_window: tuple[float, float] | None = None,
    feedback_pool_window: tuple[float, float] | None = None,
) -> str:
    """Entry point executed locally or by submitit on the cluster."""
    deriv = Path(derivatives_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out = run_decoding_for_quantity_and_epoch_benchmark(
        DEFAULT_SUBJECT_INFO,
        deriv,
        quantity,
        epoch_type,
        trial_type,
        sensor_type,
        baseline,
        lowpass,
        MODEL_NAME=model_name,
        pooled_decoding=pooled_decoding,
        response_pool_window=response_pool_window,
        feedback_pool_window=feedback_pool_window,
    )

    saved_path = _save_job_output(
        out_dir,
        quantity,
        epoch_type,
        trial_type,
        sensor_type,
        baseline,
        lowpass,
        out,
    )

    return str(saved_path)


def _prepare_subject_data(
    *,
    subject: str,
    sessions: dict,
    derivatives_dir: Path,
    quantity: str,
    epoch_type: str,
    trial_type: str,
    sensor_type: str,
    baseline: tuple[float, float] | None,
    lowpass: float,
):
    subject_epochs = []
    subject_metadata = []
    n = -1

    for session, _runs in sessions.items():
        n += 1
        io_df = load_model_metadata(subject, session, derivatives_dir, DEFAULT_SUBJECT_INFO)
        io_df["run"] = io_df["run"] + n * 8 - 1

        epochs = load_epochs(subject, session, derivatives_dir)
        if epochs is None or epoch_type not in epochs.event_id:
            continue

        event = epochs[epoch_type].copy().filter(None, lowpass).apply_baseline(baseline)
        subject_epochs.append(event)
        subject_metadata.append(io_df)

    if not subject_epochs:
        return None

    concatenated_epochs = mne.concatenate_epochs(subject_epochs, on_mismatch="ignore")

    import pandas as pd

    concatenated_metadata = pd.concat(subject_metadata, ignore_index=True)

    concatenated_epochs.metadata = concatenated_metadata
    if trial_type == "left":
        concatenated_epochs = concatenated_epochs['arm_choice=="A"']
        concatenated_metadata = concatenated_metadata[concatenated_metadata["arm_choice"] == "A"]
    elif trial_type == "right":
        concatenated_epochs = concatenated_epochs['arm_choice=="B"']
        concatenated_metadata = concatenated_metadata[concatenated_metadata["arm_choice"] == "B"]
    elif trial_type == "forced":
        concatenated_epochs = concatenated_epochs["forced.notna()"]
        concatenated_metadata = concatenated_metadata[concatenated_metadata["forced"].notna()]
    elif trial_type == "free":
        concatenated_epochs = concatenated_epochs["forced.isna()"]
        concatenated_metadata = concatenated_metadata[concatenated_metadata["forced"].isna()]
    elif trial_type == "repeat":
        concatenated_epochs = concatenated_epochs["repeat==1"]
        concatenated_metadata = concatenated_metadata[concatenated_metadata["repeat"] == 1]
    elif trial_type == "switch":
        concatenated_epochs = concatenated_epochs["repeat==0"]
        concatenated_metadata = concatenated_metadata[concatenated_metadata["repeat"] == 0]
    elif trial_type == "free_repeat":
        concatenated_epochs = concatenated_epochs["forced.isna() and repeat==1"]
        concatenated_metadata = concatenated_metadata.query("forced.isna() and repeat == 1")
    elif trial_type == "free_switch":
        concatenated_epochs = concatenated_epochs["forced.isna() and repeat==0"]
        concatenated_metadata = concatenated_metadata.query("forced.isna() and repeat == 0")

    if quantity not in concatenated_metadata.columns:
        return None

    y = concatenated_metadata[quantity].to_numpy()
    if sensor_type == "mag":
        epochs_meg = concatenated_epochs.copy().pick_types(meg="mag")
    elif sensor_type == "grad":
        epochs_meg = concatenated_epochs.copy().pick_types(meg="grad")
    else:
        epochs_meg = concatenated_epochs.copy().pick_types(meg=True)

    X = epochs_meg.get_data()
    runs = np.asarray(concatenated_metadata["run"])

    mask = np.isfinite(y)
    X = X[mask]
    y = y[mask]
    runs = runs[mask]
    if len(y) == 0:
        return None

    return X, y, runs, concatenated_epochs.times


def _compute_shared_null_model(
    *,
    derivatives_dir: Path,
    quantity: str,
    epoch_type: str,
    trial_type: str,
    sensor_type: str,
    baseline: tuple[float, float] | None,
    lowpass: float,
    model_name: str | None,
    pooled_decoding: bool,
    response_pool_window: tuple[float, float] | None,
    feedback_pool_window: tuple[float, float] | None,
    n_permutations: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    all_null_corrs = []
    subjects_included = []
    times = None

    for subject, sessions in DEFAULT_SUBJECT_INFO.items():
        prepared = _prepare_subject_data(
            subject=subject,
            sessions=sessions,
            derivatives_dir=derivatives_dir,
            quantity=quantity,
            epoch_type=epoch_type,
            trial_type=trial_type,
            sensor_type=sensor_type,
            baseline=baseline,
            lowpass=lowpass,
        )
        if prepared is None:
            continue

        X, y, runs, times = prepared
        if pooled_decoding:
            X_windowed = apply_pooling_window(
                X,
                times,
                epoch_type,
                response_pool_window=response_pool_window,
                feedback_pool_window=feedback_pool_window,
            )
            X_eff = X_windowed.mean(axis=2)
            n_features_out = 1
        else:
            X_eff = X
            n_features_out = X.shape[2]

        splits = list(LeaveOneGroupOut().split(X_eff, y, groups=runs))

        rng = np.random.RandomState(random_seed + int(subject))
        subject_null_corrs = np.zeros((n_permutations, n_features_out), dtype=float)

        for p in range(n_permutations):
            fold_corrs = np.zeros((len(splits), n_features_out), dtype=float)
            for i, (tr, te) in enumerate(splits):
                reg = get_model(model_name=model_name, alphas=None, inner_cv_splits=5)
                y_tr_perm = np.asarray(y[tr]).copy()
                rng.shuffle(y_tr_perm)

                if pooled_decoding:
                    reg.fit(X_eff[tr], y_tr_perm)
                    y_pred = reg.predict(X_eff[te])
                    corr_val = np.corrcoef(y[te], y_pred)[0, 1]
                    fold_corrs[i, 0] = corr_val
                else:
                    time_decoder = SlidingEstimator(reg, scoring=None)
                    time_decoder.fit(X_eff[tr], y_tr_perm)
                    y_pred = time_decoder.predict(X_eff[te])
                    fold_corrs[i, :] = corr_per_time(y[te], y_pred)

            subject_null_corrs[p, :] = fold_corrs.mean(axis=0)

        all_null_corrs.append(subject_null_corrs)
        subjects_included.append(subject)

    if all_null_corrs:
        if pooled_decoding:
            return np.asarray(all_null_corrs, dtype=float), np.asarray([np.nan], dtype=float), subjects_included
        return np.asarray(all_null_corrs, dtype=float), times, subjects_included
    return np.empty((0, n_permutations, 0), dtype=float), times, subjects_included


def run_shared_null_model_job(
    *,
    derivatives_dir: str,
    output_dir: str,
    quantity: str,
    epoch_type: str,
    trial_type: str,
    sensor_type: str,
    baseline: tuple[float, float] | None,
    lowpass: float,
    model_name: str | None,
    pooled_decoding: bool = False,
    response_pool_window: tuple[float, float] | None = None,
    feedback_pool_window: tuple[float, float] | None = None,
    n_permutations: int = DEFAULT_NULL_PERMUTATIONS,
    random_seed: int = DEFAULT_NULL_RANDOM_SEED,
) -> str:
    deriv = Path(derivatives_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    null_corrs, times, subjects_included = _compute_shared_null_model(
        derivatives_dir=deriv,
        quantity=quantity,
        epoch_type=epoch_type,
        trial_type=trial_type,
        sensor_type=sensor_type,
        baseline=baseline,
        lowpass=lowpass,
        model_name=model_name,
        pooled_decoding=pooled_decoding,
        response_pool_window=response_pool_window,
        feedback_pool_window=feedback_pool_window,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )

    out_npz = out_dir / "null_model_correlation.npz"
    if times is not None:
        np.savez_compressed(
            out_npz,
            null_corrs,
            times_s=np.asarray(times, dtype=float),
            subjects=np.asarray(subjects_included, dtype=str),
        )
    else:
        np.savez_compressed(
            out_npz,
            null_corrs,
            subjects=np.asarray(subjects_included, dtype=str),
        )
    return str(out_npz)


def run_single_decoding_job_from_dict(job_cfg: dict) -> str:
    """Small wrapper so submitit can map over a list of job dictionaries."""
    return run_single_decoding_job(**job_cfg)


def run_cluster_job(job_cfg: dict) -> str:
    job_type = job_cfg["job_type"]
    payload = job_cfg["payload"]
    if job_type == "decoding":
        return run_single_decoding_job(**payload)
    if job_type == "null_model":
        return run_shared_null_model_job(**payload)
    raise ValueError(f"Unknown job_type: {job_type}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit MEG decoding jobs with submitit")
    # parser.add_argument(
    #     "--derivatives-dir",
    #     type=Path,
    #     default=Path(__file__).resolve().parent / "derivatives",
    #     help="Path to derivatives directory",
    # )
    # parser.add_argument(
    #     "--output-dir",
    #     type=Path,
    #     default=None,
    #     help="Output directory for decoding .npz files",
    # )
    parser.add_argument(
        "--model-name",
        type=str,
        default="none",
        help="Model name from decoding utils: none|ridge|linear|svr|rf",
    )
    parser.add_argument("--trial-type", type=str, default="free")
    parser.add_argument("--sensor-type", type=str, default="mag")
    parser.add_argument("--baseline", type=str, default="none")
    parser.add_argument("--lowpass", type=float, default=15.0)
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
    parser.add_argument(
        "--pooled-decoding",
        "--pooling",
        dest="pooled_decoding",
        action="store_true",
        help="Use pooled decoding (single model over time-pooled features)",
    )

    parser.add_argument(
        "--job",
        action="append",
        default=[],
        help=(
            "Job specification QUANTITY:EPOCH_TYPE. "
            "Repeat this flag to submit multiple jobs. "
            "If omitted, defaults to ER/EU response and PE feedback."
        ),
    )
    parser.add_argument(
        "--null-only",
        action="store_true",
        help="Submit only the shared null-model job (skip decoding jobs)",
    )
    parser.add_argument(
        "--run-local",
        action="store_true",
        help="Run jobs in-process (debug mode) instead of submitting to SLURM",
    )

    parser.add_argument("--logs-dir", type=Path, default=Path("submitit_logs"))
    parser.add_argument("--job-name", type=str, default="meg-decoding")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    derivatives_dir = Path(__file__).resolve().parent / "derivatives"
    model_name = None if args.model_name.lower() in {"none", "null"} else args.model_name
    baseline = _parse_baseline(args.baseline)
    response_pool_window = _parse_pool_window(args.response_pool_window, "--response-pool-window")
    feedback_pool_window = _parse_pool_window(args.feedback_pool_window, "--feedback-pool-window")

    model_tag = model_name if model_name is not None else "RidgeCV"
    decode_mode = "pooled" if args.pooled_decoding else "time_resolved"
    output_dir = derivatives_dir.parent / "decoding_outputs" / f"model_{model_tag}" / decode_mode
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = _parse_jobs(args.job) if args.job else list(DEFAULT_DECODING_JOBS)
    job_specs = []

    if not args.null_only:
        job_specs.extend(
            [
                {
                    "job_type": "decoding",
                    "payload": {
                        "derivatives_dir": str(derivatives_dir),
                        "output_dir": str(output_dir),
                        "quantity": quantity,
                        "epoch_type": epoch_type,
                        "trial_type": args.trial_type,
                        "sensor_type": args.sensor_type,
                        "baseline": baseline,
                        "lowpass": args.lowpass,
                        "model_name": model_name,
                        "pooled_decoding": args.pooled_decoding,
                        "response_pool_window": response_pool_window,
                        "feedback_pool_window": feedback_pool_window,
                    },
                }
                for quantity, epoch_type in jobs
            ]
        )

    null_quantity, null_epoch = DEFAULT_NULL_REFERENCE_JOB
    job_specs.append(
        {
            "job_type": "null_model",
            "payload": {
                "derivatives_dir": str(derivatives_dir),
                "output_dir": str(output_dir),
                "quantity": null_quantity,
                "epoch_type": null_epoch,
                "trial_type": args.trial_type,
                "sensor_type": args.sensor_type,
                "baseline": baseline,
                "lowpass": args.lowpass,
                "model_name": model_name,
                "pooled_decoding": args.pooled_decoding,
                "response_pool_window": response_pool_window,
                "feedback_pool_window": feedback_pool_window,
                "n_permutations": DEFAULT_NULL_PERMUTATIONS,
                "random_seed": DEFAULT_NULL_RANDOM_SEED,
            },
        }
    )

    if args.run_local:
        print("Running locally (no cluster submission).")
        for spec in job_specs:
            saved_path = run_cluster_job(spec)
            payload = spec["payload"]
            if spec["job_type"] == "decoding":
                print(
                    f"Completed {payload['quantity']}:{payload['epoch_type']} "
                    f"-> {saved_path}"
                )
            else:
                print(f"Completed shared null model -> {saved_path}")
        return

    logs_dir = args.logs_dir.resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    executor = submitit.AutoExecutor(folder=str(logs_dir))
    executor.update_parameters(name=args.job_name, **CLUSTER_SUBMITIT_CONFIG)

    submitted_jobs = executor.map_array(run_cluster_job, job_specs)
    print(f"Submitted {len(submitted_jobs)} jobs.")
    for spec, job in zip(job_specs, submitted_jobs):
        payload = spec["payload"]
        if spec["job_type"] == "decoding":
            print(
                f"- {payload['quantity']}:{payload['epoch_type']} -> "
                f"job_id={job.job_id} output_dir={output_dir}"
            )
        else:
            print(f"- shared-null-model -> job_id={job.job_id} output_dir={output_dir}")


if __name__ == "__main__":
    main()