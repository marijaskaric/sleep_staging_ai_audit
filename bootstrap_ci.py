"""
Subject-level (cluster) bootstrap confidence intervals
=======================================================
The 142 recordings come from 78 subjects (64 recorded on two nights).
A normal-approximation CI with n=142 treats the two nights of the same
person as independent and is too narrow. Here we resample SUBJECTS (with
both of their nights) instead of recordings, and report percentile 95%
CIs for the mean of each metric across recordings.

Requires yasa_sleepedfx_pilot_results.csv and
yasa_sleepedfx_clinical_stats.csv (from yasa_sleepedfx_pilot.py).
Writes Sleep Cassette/results/bootstrap_ci.csv.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd

RESULTS_DIR = "C:/Users/karic/Documents/Mara/Projekat/Sleep Cassette/results"
N_BOOT = 10_000
SEED = 20261002  # fixed so the CIs are reproducible

results = pd.read_csv(os.path.join(RESULTS_DIR, "yasa_sleepedfx_pilot_results.csv"))
clinical = pd.read_csv(os.path.join(RESULTS_DIR, "yasa_sleepedfx_clinical_stats.csv"))
df = results.merge(clinical, on="subject")
df["subject_num"] = df["subject"].str[3:5].astype(int)

metrics = ["accuracy", "kappa", "macro_f1",
           "TST_diff_min", "SE_diff_pct", "REM_lat_diff_min", "WASO_diff_min"]

# Per-subject sums and recording counts, so each bootstrap replicate is a
# cheap weighted sum instead of a concat of resampled rows.
by_subj = df.groupby("subject_num")
sums = by_subj[metrics].sum().to_numpy()
counts = by_subj.size().to_numpy()
n_subj = len(counts)

rng = np.random.default_rng(SEED)
idx = rng.integers(0, n_subj, size=(N_BOOT, n_subj))
weights = np.zeros((N_BOOT, n_subj))
np.add.at(weights, (np.arange(N_BOOT)[:, None], idx), 1)
boot_means = (weights @ sums) / (weights @ counts)[:, None]

lo, hi = np.percentile(boot_means, [2.5, 97.5], axis=0)
naive_se = df[metrics].std() / np.sqrt(len(df))
out = pd.DataFrame({
    "metric": metrics,
    "mean": df[metrics].mean().to_numpy(),
    "ci95_low": lo,
    "ci95_high": hi,
    "naive_ci95_low": (df[metrics].mean() - 1.96 * naive_se).to_numpy(),
    "naive_ci95_high": (df[metrics].mean() + 1.96 * naive_se).to_numpy(),
})

print(f"{len(df)} recordings from {n_subj} subjects, {N_BOOT} subject-level resamples (seed {SEED})")
print(out.round(4).to_string(index=False))
out.to_csv(os.path.join(RESULTS_DIR, "bootstrap_ci.csv"), index=False)
print(f"\nSaved {os.path.join(RESULTS_DIR, 'bootstrap_ci.csv')}")
