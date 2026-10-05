"""
REM latency and worst-case recording check
===========================================
YASA's REM latency is ~69 min shorter than the human one on average. This
script checks where that comes from, using the per-epoch hypnograms that
yasa_sleepedfx_pilot.py saves (yasa_sleepedfx_epochs.csv):

  1. Which stage YASA assigns to the epochs the human scored as WAKE
     before the human sleep onset (in bed, lights off, not yet asleep).
  2. What YASA's first "sleep" epoch is, and how often its REM latency
     is ~0 (first REM within 5 min of its own sleep onset).
  3. For the three recordings with the lowest accuracy: stage confusion
     matrix (human rows x YASA columns) and stage proportions.

Writes Sleep Cassette/results/rem_latency_check.txt.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd

RESULTS_DIR = "C:/Users/karic/Documents/Mara/Projekat/Sleep Cassette/results"
STAGES = ["WAKE", "N1", "N2", "N3", "REM"]
SLEEP = {"N1", "N2", "N3", "REM"}
EPOCH_MIN = 0.5

epochs = pd.read_csv(os.path.join(RESULTS_DIR, "yasa_sleepedfx_epochs.csv"))
results = pd.read_csv(os.path.join(RESULTS_DIR, "yasa_sleepedfx_pilot_results.csv"))
lines = []


def out(text=""):
    print(text)
    lines.append(str(text))


pre_onset = []   # (human, yasa) pairs for epochs before human sleep onset
rows = []
for sid, g in epochs.groupby("subject", sort=False):
    human, yasa = g["human"].to_numpy(dtype=object), g["yasa"].to_numpy(dtype=object)
    h_sleep = np.where(np.isin(human, list(SLEEP)))[0]
    y_sleep = np.where(np.isin(yasa, list(SLEEP)))[0]
    h_on, y_on = h_sleep[0], y_sleep[0]
    y_rem = np.where(yasa == "REM")[0]
    y_rem_after = y_rem[y_rem >= y_on]
    pre_onset.append(pd.DataFrame({"human": human[:h_on], "yasa": yasa[:h_on]}))
    rows.append({
        "subject": sid,
        "human_onset_min": h_on * EPOCH_MIN,
        "yasa_onset_min": y_on * EPOCH_MIN,
        "yasa_first_sleep_stage": yasa[y_on],
        "yasa_rem_lat_min": (y_rem_after[0] - y_on) * EPOCH_MIN if len(y_rem_after) else np.nan,
        "yasa_rem_before_human_onset": int(np.sum(yasa[:h_on] == "REM")),
    })
rec = pd.DataFrame(rows)
pre = pd.concat(pre_onset)

out("=== 1. YASA labels for human-WAKE epochs before human sleep onset ===")
pre_wake = pre[pre["human"] == "WAKE"]
out(f"{len(pre_wake)} epochs ({len(pre_wake) * EPOCH_MIN / 60:.1f} h) across {len(rec)} recordings")
out((pre_wake["yasa"].value_counts(normalize=True).reindex(STAGES).fillna(0) * 100).round(1).to_string())

out("\n=== 2. YASA sleep onset and REM latency ===")
out(f"YASA sleep onset earlier than human: {(rec.yasa_onset_min < rec.human_onset_min).sum()} / {len(rec)} recordings "
    f"(median {np.median(rec.human_onset_min - rec.yasa_onset_min):.1f} min earlier)")
out("YASA's first sleep epoch is:")
out(rec["yasa_first_sleep_stage"].value_counts().to_string())
out(f"YASA REM latency < 5 min: {(rec.yasa_rem_lat_min < 5).sum()} / {len(rec)} recordings")
out(f"Recordings with >=1 YASA REM epoch before human sleep onset: {(rec.yasa_rem_before_human_onset > 0).sum()} / {len(rec)}")

out("\n=== 3. Worst-case recordings (lowest accuracy) ===")
for sid in results.nsmallest(3, "accuracy")["subject"]:
    g = epochs[(epochs.subject == sid) & epochs.human.notna()]
    acc = results.loc[results.subject == sid, "accuracy"].iloc[0]
    out(f"\n--- {sid}: accuracy {acc:.3f}, {len(g)} scored epochs ---")
    cm = pd.crosstab(g["human"], g["yasa"]).reindex(index=STAGES, columns=STAGES, fill_value=0)
    cm.index.name, cm.columns.name = "human \\ YASA", None
    out(cm.to_string())
    prop = pd.DataFrame({"human_%": g["human"].value_counts(normalize=True),
                         "yasa_%": g["yasa"].value_counts(normalize=True)}).reindex(STAGES).fillna(0) * 100
    out(prop.round(1).to_string())

with open(os.path.join(RESULTS_DIR, "rem_latency_check.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nSaved {os.path.join(RESULTS_DIR, 'rem_latency_check.txt')}")
