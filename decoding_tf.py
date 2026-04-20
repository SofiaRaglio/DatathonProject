#!/usr/bin/env python3
"""
Time-Frequency MEG Decoding - Combined Computation and Analysis Script

This script performs both the computational part (time-frequency decoding of cognitive variables 
from MEG signals) and the comparison/plotting part (comparison with original time-domain decoding).

Time-Frequency parameters are set in this script:
- Frequency band
- Features
- Time window
- PCA components

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
from decoding_notebook_utils import run_decoding_for_quantity_and_epoch, TupleKeyEncoder

# Configuration
DERIVATIVES_DIR = Path('/home/fmeyniel/nasShare/projects/EXPLORE_PLUS/DatathonProject/derivatives')
OUTPUT_DIR = Path('/home/fmeyniel/projects/DatathonProject/decoding_outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Subject information
SUBJECT_INFO = {'01': {'4': [1,2,3,4,5,6,7,8], '5': [1,2,3,4,5,6,7,8]},
                '11': {'4': [1,2,3,4,5,6,7,8], '5': [1,2,3,4,5,6,7,8]},
                '19': {'2': [1,2,3,4,5,6,7,8], '3': [1,2,3,4,5,6,7,8]},
}
SUBJECT_IDS = list(SUBJECT_INFO.keys())

# Time-frequency configuration
TF_CONFIG = {
        'FREQ_BANDS': [(0.5, 2),
                       (2, 4),
                       (4, 8),
                       (8, 12),
                       (12, 40)],
        'feature_types': [['real', 'imag'],
                          ['real', 'imag'],
                          ['real', 'imag'],
                          ['amplitude'],
                          ['amplitude']],
        'time_window': (-0.5, 0.75),
        'pca': {'amplitude': 40,
                ('real', 'imag'): 80}
    }

# Decoding jobs - same as original notebook
DECODING_JOBS = [
    ("ER_diffRS_z", "response"),      # Expected Reward at response
    ("EU_diffRS_z", "response"),      # Estimated Uncertainty at response
    ("PE_z", "feedback"),             # Prediction Error at feedback
]

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
    "subjects": SUBJECT_IDS,
}
with open(analysis_dir / 'config.json', 'w') as f:
    json.dump(config, f, indent=2, cls=TupleKeyEncoder)
print(f"Configuration saved to {analysis_dir / 'config.json'}")

# ===== DECODING =====
print("\n=== DECODING ===")

# Run time-frequency decoding
tf_results = {}

for quantity, epoch_type in DECODING_JOBS:
    print(f"Running TF decoding for {quantity} at {epoch_type}...")
    
    result = run_decoding_for_quantity_and_epoch(
        SUBJECT_INFO, DERIVATIVES_DIR, quantity, epoch_type,
        trial_type='free', sensor_type='mag', baseline=None, LOWPASS=None,
        time_freq_options=TF_CONFIG
    )

    # Create flat structure matching original NPZ format
    flat_result = {'times_s': result['times']}
    for i, sub in enumerate(SUBJECT_IDS):
        key = f'corr_r_sub{sub}'
        flat_result[key] = result['all_corrs'][i] if len(result['all_corrs']) > i else None
    
    # Save as NPZ file with same naming convention as original
    filename = f'{quantity}_{epoch_type}_tf_free_mag.npz'
    np.savez(analysis_dir / filename, **flat_result)
    print(f"Saved {filename}")
    
    tf_results[(quantity, epoch_type)] = flat_result
    print(f"Completed {quantity} at {epoch_type}")

print(f"TF results saved to {analysis_dir}")
print(f"\nComputation completed successfully!")

# ===== COMPARISON/PLOTTING =====
print("\n=== COMPARISON/PLOTTING ===")
print("=== Implement comparison with original results ===")

# Load original results
null_model = np.load(OUTPUT_DIR / 'null_model_correlation.npz')
original_results = {}
for quantity, epoch_type in DECODING_JOBS:
    filename = f'{quantity}_{epoch_type}_free_mag_None_15.npz'
    original_results[(quantity, epoch_type)] = np.load(OUTPUT_DIR / filename)
print("Original results loaded successfully")

# Create comparison plots
print("\nCreating comparison plots...")

fig, axes = plt.subplots(len(DECODING_JOBS), 1, figsize=(10, 12), sharex=True)
titles = [f'{quantity} at {epoch_type}'
          for quantity, epoch_type in DECODING_JOBS]

for i, (ax, title) in enumerate(zip(axes, titles)):
    quantity, epoch_type = DECODING_JOBS[i]
    
    # Get time points
    tf_times = tf_results[(quantity, epoch_type)]['times_s']
    orig_times = original_results[(quantity, epoch_type)]['times_s']
    
    # Plot null model
    null_mean = np.mean(np.mean(null_model['arr_0'], axis=1), axis=0)
    ax.plot(orig_times, null_mean, color='k', alpha=0.85, linestyle=':', label='Null model')
    
    # Plot results for each subject
    for j, sid in enumerate(SUBJECT_IDS):
        # Time-frequency results (plain lines)
        tf_key = f'corr_r_sub{sid}'
        if tf_key in tf_results[(quantity, epoch_type)]:
            tf_corr = tf_results[(quantity, epoch_type)][tf_key]
            ax.plot(tf_times, tf_corr, color=f'C{j}', linestyle='-',
                    label=f'TF sub-{sid}', linewidth=2)
        
        # Original results (dashed lines)
        orig_key = f'corr_r_sub{sid}'
        if orig_key in original_results[(quantity, epoch_type)]:
            orig_corr = original_results[(quantity, epoch_type)][orig_key]
            ax.plot(orig_times, orig_corr, color=f'C{j}', linestyle='-', alpha=0.5,
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
    tf_times = tf_results[(quantity, epoch_type)]['times_s']
    orig_times = original_results[(quantity, epoch_type)]['times_s']
    
    tf_mask = (tf_times >= STAT_WINDOW[epoch_type][0]) & (tf_times <= STAT_WINDOW[epoch_type][1])
    orig_mask = (orig_times >= STAT_WINDOW[epoch_type][0]) & (orig_times <= STAT_WINDOW[epoch_type][1])
    
    # Calculate statistics for each subject
    for sid in SUBJECT_IDS:
        if sid not in stats:
            stats[sid] = {}
        
        # TF method statistics
        tf_key = f'corr_r_sub{sid}'
        if tf_key in tf_results[(quantity, epoch_type)]:
            tf_corr = tf_results[(quantity, epoch_type)][tf_key]
            stats[sid][f'{quantity}_tf'] = np.mean(tf_corr[tf_mask])
        
        # Original method statistics
        orig_key = f'corr_r_sub{sid}'
        if orig_key in original_results[(quantity, epoch_type)]:
            orig_corr = original_results[(quantity, epoch_type)][orig_key]
            stats[sid][f'{quantity}_orig'] = np.mean(orig_corr[orig_mask])

stats_df = pd.DataFrame(stats).T

print("\n Summary Statistics (mean correlation in event window):")
print(stats_df)

# Save statistics to CSV
stats_df.to_csv(analysis_dir / 'tf_vs_original_statistics.csv')
print(f"\n Statistics saved to {analysis_dir / 'tf_vs_original_statistics.csv'}")

print(f"\n=== ALL PROCESSING COMPLETED ===")
