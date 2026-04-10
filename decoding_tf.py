#!/usr/bin/env python3
"""
Time-Frequency MEG Decoding - Combined Computation and Analysis Script

This script performs both the computational part (time-frequency decoding of cognitive variables 
from MEG signals) and the analysis/plotting part (comparison with original time-domain decoding).

Time-Frequency Parameters:
- Frequency band: (2, 4) Hz (theta band)
- Features: Real and imaginary parts
- Time window: (-0.5, 0.75) seconds
- PCA components: 50

Output:
- Creates timestamped directory with configuration, TF results, and analysis outputs
- Saves both raw results and comparison plots
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
from decoding_notebook_utils import run_decoding_for_quantity_and_epoch

# Configuration
DERIVATIVES_DIR = Path('/home/fmeyniel/nasShare/projects/EXPLORE_PLUS/DatathonProject/derivatives')
OUTPUT_DIR = Path('/home/fmeyniel/projects/DatathonProject/decoding_outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Subject information
SUBJECT_INFO = {'01': {'4': [1,2,3,4,5,6,7,8], '5': [1,2,3,4,5,6,7,8]},
                '11': {'4': [1,2,3,4,5,6,7,8], '5': [1,2,3,4,5,6,7,8]},
                '19': {'2': [1,2,3,4,5,6,7,8], '3': [1,2,3,4,5,6,7,8]},
}

# Time-frequency configuration
TF_CONFIG = {
    'FREQ_BANDS': [(0.5, 8)],
    'feature_types': ['real', 'imag'],  # Use real and imaginary parts
    'time_window': (-0.5, 0.75),        # Extended time window
    'pca': 50                           # Number of principal components
}

# Decoding jobs - same as original notebook
DECODING_JOBS = [
    ("ER_diffRS_z", "response"),      # Expected Reward at response
    ("EU_diffRS_z", "response"),      # Estimated Uncertainty at response
    ("PE_z", "feedback"),            # Prediction Error at feedback
]

# Subject colors for consistent plotting
SUBJECT_IDS = ['01', '11', '19']

# Statistical analysis windows
STAT_WINDOW = {
    'feedback': (0.2, 0.5),
    'response': (-0.2, 0.2),
}

print("Starting time-frequency decoding computation and analysis...")

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
analysis_dir = OUTPUT_DIR / f"tf_analysis_{timestamp}"
analysis_dir.mkdir(parents=True, exist_ok=True)
print(f"Created output directory: {analysis_dir}")

# Save configuration as JSON
config = {
    "timestamp": timestamp,
    "analysis_type": "time_frequency_comparison",
    "time_frequency_config": TF_CONFIG,
    "decoding_jobs": DECODING_JOBS,
    "subjects": list(SUBJECT_INFO.keys()),
}

with open(analysis_dir / 'config.json', 'w') as f:
    json.dump(config, f, indent=2)
print(f"Configuration saved to {analysis_dir / 'config.json'}")

# ===== COMPUTATION PHASE =====
print("\n=== COMPUTATION PHASE ===")

# Run time-frequency decoding
tf_results = {}

for quantity, epoch_type in DECODING_JOBS:
    print(f"Running TF decoding for {quantity} at {epoch_type}...")
    
    result = run_decoding_for_quantity_and_epoch(
        SUBJECT_INFO, DERIVATIVES_DIR, quantity, epoch_type,
        trial_type='free', sensor_type='mag', baseline=None, LOWPASS=None,
        time_freq_options=TF_CONFIG
    )
    
    tf_results[(quantity, epoch_type)] = result
    print(f"Completed {quantity} at {epoch_type}")

# Restructure results to match original format and save individual NPZ files
print("\nSaving time-frequency results in original format...")

for (quantity, epoch_type), result in tf_results.items():
    # Create flat structure matching original NPZ format
    flat_result = {'times_s': result['times']}
    for i, sub in enumerate(SUBJECT_IDS):
        key = f'corr_r_sub{sub}'
        flat_result[key] = result['all_corrs'][i] if len(result['all_corrs']) > i else None
    
    # Save as NPZ file with same naming convention as original
    filename = f'{quantity}_{epoch_type}_tf_free_mag.npz'
    np.savez(analysis_dir / filename, **flat_result)
    print(f"Saved {filename}")

# Save combined results
print("\nSaving combined results...")
np.save(analysis_dir / 'tf_results.npy', tf_results, allow_pickle=True)
print(f"TF results saved to {analysis_dir / 'tf_results.npy'}")

print(f"\nComputation completed successfully!")

# ===== ANALYSIS/PLOTTING PHASE =====
print("\n=== ANALYSIS/PLOTTING PHASE ===")

# Load TF results (freshly computed)
tf_results_loaded = {}
ER_tf = np.load(analysis_dir / 'ER_diffRS_z_response_tf_free_mag.npz')
EU_tf = np.load(analysis_dir / 'EU_diffRS_z_response_tf_free_mag.npz')
PE_tf = np.load(analysis_dir / 'PE_z_feedback_tf_free_mag.npz')

tf_results_loaded[('ER_diffRS_z', 'response')] = ER_tf
tf_results_loaded[('EU_diffRS_z', 'response')] = EU_tf
tf_results_loaded[('PE_z', 'feedback')] = PE_tf

print(f"TF results loaded from {analysis_dir}")

# Load original results
original_results = {}

ER_original = np.load(OUTPUT_DIR / 'ER_diffRS_z_response_free_mag_None_15.npz')
EU_original = np.load(OUTPUT_DIR / 'EU_diffRS_z_response_free_mag_None_15.npz')
PE_original = np.load(OUTPUT_DIR / 'PE_z_feedback_free_mag_None_15.npz')
null_model = np.load(OUTPUT_DIR / 'null_model_correlation.npz')

original_results[('ER_diffRS_z', 'response')] = ER_original
original_results[('EU_diffRS_z', 'response')] = EU_original
original_results[('PE_z', 'feedback')] = PE_original

print("Original results loaded successfully")

# Create comparison plots
print("\nCreating comparison plots...")

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
titles = [
    'ER_diffRS_z (Expected Reward) at Response',
    'EU_diffRS_z (Estimated Uncertainty) at Response',
    'PE_z (Prediction Error) at Feedback'
]

for i, (ax, title) in enumerate(zip(axes, titles)):
    quantity, epoch_type = DECODING_JOBS[i]
    
    # Get time points
    tf_times = tf_results_loaded[(quantity, epoch_type)]['times_s']
    orig_times = original_results[(quantity, epoch_type)]['times_s']
    
    # Plot null model
    null_mean = np.mean(np.mean(null_model['arr_0'], axis=1), axis=0)
    ax.plot(orig_times, null_mean, color='k', alpha=0.85, linestyle=':', label='Null model')
    
    # Plot results for each subject
    for j, sid in enumerate(SUBJECT_IDS):
        color = f'C{j}'
        
        # Time-frequency results (plain lines)
        tf_key = f'corr_r_sub{sid}'
        if tf_key in tf_results_loaded[(quantity, epoch_type)]:
            tf_corr = tf_results_loaded[(quantity, epoch_type)][tf_key]
            ax.plot(tf_times, tf_corr, color=color, linestyle='-',
                    label=f'TF sub-{sid}', linewidth=2)
        
        # Original results (dashed lines)
        orig_key = f'corr_r_sub{sid}'
        if orig_key in original_results[(quantity, epoch_type)]:
            orig_corr = original_results[(quantity, epoch_type)][orig_key]
            ax.plot(orig_times, orig_corr, color=color, linestyle='-', alpha=0.5,
                    label=f'Orig sub-{sid}', linewidth=1)
    
    # Formatting
    ax.axhline(0.0, color='gray', lw=0.5, ls=':')
    ax.axvline(0.0, color='gray', lw=0.6, ls='--')
    ax.set_ylabel('Correlation (r)', fontsize=12)
    ax.set_title(title, fontsize=14, pad=10)
    
    if i == 2:  # Only show legend on last plot to avoid clutter
        ax.legend(loc='best', fontsize=10, framealpha=0.8)

axes[-1].set_xlabel('Time (s)', fontsize=12)
fig.tight_layout()
plt.savefig(analysis_dir / 'tf_vs_original_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Comparison plots created and saved")

# Compute statistics
print("\nCalculating summary statistics...")

stats = {}

for quantity, epoch_type in DECODING_JOBS:
    # Get time points and create mask
    tf_times = tf_results_loaded[(quantity, epoch_type)]['times_s']
    orig_times = original_results[(quantity, epoch_type)]['times_s']
    
    tf_mask = (tf_times >= STAT_WINDOW[epoch_type][0]) & (tf_times <= STAT_WINDOW[epoch_type][1])
    orig_mask = (orig_times >= STAT_WINDOW[epoch_type][0]) & (orig_times <= STAT_WINDOW[epoch_type][1])
    
    # Calculate statistics for each subject
    for sid in SUBJECT_IDS:
        if sid not in stats:
            stats[sid] = {}
        
        # TF method statistics
        tf_key = f'corr_r_sub{sid}'
        if tf_key in tf_results_loaded[(quantity, epoch_type)]:
            tf_corr = tf_results_loaded[(quantity, epoch_type)][tf_key]
            stats[sid][f'{quantity}_tf'] = np.mean(tf_corr[tf_mask])
        
        # Original method statistics
        orig_key = f'corr_r_sub{sid}'
        if orig_key in original_results[(quantity, epoch_type)]:
            orig_corr = original_results[(quantity, epoch_type)][orig_key]
            stats[sid][f'{quantity}_orig'] = np.mean(orig_corr[orig_mask])

stats_df = pd.DataFrame(stats).T

print("\n Summary Statistics (mean correlation in event window):")
print(stats_df)

# Calculate and display improvements
print("\n Performance Improvement (TF - Original):")

for quantity, epoch_type in DECODING_JOBS:
    quantity_key = quantity.split('_')[0]  # 'ER', 'EU', 'PE'
    tf_col = f'{quantity_key}_tf'
    orig_col = f'{quantity_key}_orig'
    
    # Use full column names instead of shortened versions
    tf_col = f'{quantity}_tf'
    orig_col = f'{quantity}_orig'
    
    improvement = stats_df[tf_col].mean() - stats_df[orig_col].mean()
    tf_mean = stats_df[tf_col].mean()
    orig_mean = stats_df[orig_col].mean()
    
    print(f'{quantity.split("_")[0]}: {improvement:+.3f} ' +
            f'(TF: {tf_mean:.3f}, Orig: {orig_mean:.3f})')

# Save statistics to CSV
stats_df.to_csv(analysis_dir / 'tf_vs_original_statistics.csv')
print(f"\n Statistics saved to {analysis_dir / 'tf_vs_original_statistics.csv'}")

print(f"\nAnalysis completed successfully!")

print(f"\n=== ALL PROCESSING COMPLETED ===")
print(f"\nResults saved in: {analysis_dir}")