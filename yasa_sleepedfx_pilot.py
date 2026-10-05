"""
YASA vs human scoring on Sleep-EDF Expanded (Sleep Cassette subset)
====================================================================
Main pipeline - run this first; every other script reads its outputs.

For each recording it loads the PSG and the human-scored hypnogram, runs
YASA automatic sleep staging, crops both hypnograms to the night (lights
off -> end of the main sleep period + 30 min, see SLEEP_PERIOD_MARGIN_MIN)
and compares them:
  - accuracy, Cohen's kappa, macro-F1        (standard, comparable to literature)
  - F1 per sleep stage, pooled                (surfaces the known N1 weakness)
  - clinical parameters (TST, SE, REM latency, WASO), YASA vs human
    (this mirrors what the "Beyond accuracy" and SLEEPYLAND papers
    report - a clinician cares less about per-epoch agreement than about
    "do we agree on how long the patient slept")

Outputs (in RESULTS_DIR):
  yasa_sleepedfx_pilot_results.csv   per-recording accuracy/kappa/macro-F1
  yasa_sleepedfx_clinical_stats.csv  per-recording TST/SE/REM latency/WASO
  yasa_sleepedfx_epochs.csv          per-epoch human and YASA labels
  excluded_recordings.csv            nights not analysed, with the reason

------------------------------------------------------------------------
SETUP
------------------------------------------------------------------------
1) Environment: see environment.yml (conda env `yasa_env`).

2) Data (PhysioNet, fully open):
    https://physionet.org/content/sleep-edfx/1.0.0/
   Put the Sleep Cassette EDF files in DATA_DIR and SC-subjects.xls in
   Sleep Cassette/supplementary/. Each recording has two files:
     <recording>-PSG.edf         <- raw PSG signals (EEG, EOG, EMG)
     <recording>-Hypnogram.edf   <- human scoring, as MNE annotations

   Filename pattern: SC4ssNEo-PSG.edf
     ss = subject number (00, 01, 02, ... - zero-padded, starts at 00)
     N  = night number (1 or 2)
     o  = internal recording-version letter, irrelevant here

   POPULATION NOTE: Sleep Cassette subjects are healthy Caucasian adults,
   no sleep medication, aged 25-101. There are no race/ethnicity labels,
   so results here say nothing about bias across those groups.

3) Run from the project root (paths below are relative to it).
------------------------------------------------------------------------
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # prevents an OpenMP duplicate-library
                                              # crash/hang caused by numpy+sklearn+
                                              # lightgbm each bundling their own copy

import datetime
import glob
import sys
import time

import mne
import numpy as np
import pandas as pd
import yasa
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

# ---- 1. Configuration ---------------------------------------------------

DATA_DIR = "C:/Users/karic/Documents/Mara/Projekat/Sleep Cassette/data"               # <-- adjust to your local folder
RESULTS_DIR = "C:/Users/karic/Documents/Mara/Projekat/Sleep Cassette/results"            # all output files (CSVs) are saved here
SUBJECTS_XLS = "C:/Users/karic/Documents/Mara/Projekat/Sleep Cassette/supplementary/SC-subjects.xls"  # age, sex, LightsOff per night
EEG_CHANNEL = "EEG Fpz-Cz"         # channel name as it appears in sleep-edfx files
EOG_CHANNEL = "EOG horizontal"
EMG_CHANNEL = "EMG submental"
EPOCH_SEC = 30
# Sleep Cassette recordings span ~23 h (day + night + next day). Evaluation
# is restricted to the night: from lights off (LightsOff in SC-subjects.xls)
# to the end of the human-scored main sleep period plus a margin. Without
# it, ~16 h of daytime wake inflates accuracy, and a few spurious YASA
# "sleep" epochs during the day stretch its sleep period to the whole
# recording, turning daytime wake into WASO (+482 min mean bias).
# - Lights off, not first human sleep, starts the window: in 44 recordings
#   the human scorer marked naps before lights off.
# - The main sleep period ends at the first wake gap longer than
#   MAIN_SLEEP_MAX_GAP_MIN: night-time awakenings in this data last up to
#   ~1.5 h, while longer gaps (4-7 h) lead into a nap on the next day.
SLEEP_PERIOD_MARGIN_MIN = 30
MAIN_SLEEP_MAX_GAP_MIN = 120

# Map Sleep-EDF hypnogram labels to standard AASM/YASA labels.
# N3+N4 are merged, following standard AASM 2.x convention (same as the
# original YASA validation paper does).
#
# IMPORTANT: YASA 0.7.0+ uses "WAKE"/"REM" as canonical labels (not the
# older "W"/"R"). This must match exactly, otherwise Wake and REM epochs
# silently never match between ground truth and prediction, and accuracy
# for those two stages looks artificially near zero.
STAGE_MAP = {
    "Sleep stage W": "WAKE",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "Sleep stage ?": None,   # unscored -> excluded from agreement metrics
    "Movement time": None,   # movement artifact -> excluded from agreement metrics
}

STAGES = ["WAKE", "N1", "N2", "N3", "REM"]


def log(msg):
    """Timestamped print that flushes immediately, so progress is visible
    live instead of appearing all at once at the end. Cheap to have, and
    invaluable the moment something runs slower than expected."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()


def load_recording(psg_path: str, hyp_path: str) -> mne.io.Raw:
    """Load the PSG signal file and attach the human-scored hypnogram
    to it as MNE annotations."""
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    annotations = mne.read_annotations(hyp_path)
    raw.set_annotations(annotations, emit_warning=False)
    return raw


def ground_truth_hypnogram(raw: mne.io.Raw, epoch_sec: int = EPOCH_SEC) -> np.ndarray:
    """Expand the annotation intervals (which can span many epochs) into
    one label per 30-second epoch, so the result is directly comparable
    to YASA's output, which is always one label per epoch."""
    n_epochs = int(raw.times[-1] // epoch_sec)
    labels = np.array([None] * n_epochs, dtype=object)

    for annot in raw.annotations:
        stage = STAGE_MAP.get(annot["description"])
        if stage is None:
            continue
        start_epoch = int(annot["onset"] // epoch_sec)
        end_epoch = int((annot["onset"] + annot["duration"]) // epoch_sec)
        labels[start_epoch:end_epoch] = stage

    return labels


def compute_sleep_stats(stages, epoch_sec=EPOCH_SEC):
    """Compute the clinically-relevant summary parameters from a sequence
    of per-epoch stage labels:

      - TST  (Total Sleep Time):      minutes actually spent asleep
      - SE   (Sleep Efficiency):      TST / time in bed, as a percentage
      - REM latency:                  minutes from sleep onset to first REM epoch
      - WASO (Wake After Sleep Onset): minutes of wake BETWEEN sleep onset
                                        and the last sleep epoch (excludes
                                        wake before/after the sleep period)

    These are the same kind of derived measures used in the "Beyond
    accuracy" and SLEEPYLAND papers to compare an algorithm against
    human scoring - a clinician cares less about "accuracy per epoch"
    and more about "do we agree on how long the patient slept."
    """
    stages = list(stages)
    n = len(stages)
    epoch_min = epoch_sec / 60
    sleep_stages = {"N1", "N2", "N3", "REM"}
    is_sleep = [s in sleep_stages for s in stages]

    if not any(is_sleep):
        return {"TST_min": 0.0, "SE_pct": 0.0, "REM_latency_min": np.nan, "WASO_min": 0.0}

    onset_idx = is_sleep.index(True)
    offset_idx = len(is_sleep) - 1 - is_sleep[::-1].index(True)

    TST = sum(is_sleep) * epoch_min
    TIB = n * epoch_min  # Time In Bed = evaluated window (lights off -> main sleep end + margin)
    SE = 100 * TST / TIB if TIB > 0 else np.nan

    rem_after_onset = [i for i, s in enumerate(stages) if s == "REM" and i >= onset_idx]
    REM_latency = (rem_after_onset[0] - onset_idx) * epoch_min if rem_after_onset else np.nan

    waso_epochs = sum(1 for i in range(onset_idx, offset_idx + 1) if stages[i] == "WAKE")
    WASO = waso_epochs * epoch_min

    return {"TST_min": TST, "SE_pct": SE, "REM_latency_min": REM_latency, "WASO_min": WASO}


def lights_off_epoch(raw: mne.io.Raw, lights_off: datetime.time, epoch_sec: int = EPOCH_SEC) -> int:
    """Epoch index of lights off. LightsOff is a clock time; recordings
    start in the afternoon, so a clock time earlier than the start time
    belongs to the next day (after midnight)."""
    start = raw.info["meas_date"].replace(tzinfo=None)
    lo = datetime.datetime.combine(start.date(), lights_off)
    if lo < start:
        lo += datetime.timedelta(days=1)
    return int((lo - start).total_seconds() // epoch_sec)


def analysis_window(y_true, lo_epoch: int, epoch_sec: int = EPOCH_SEC):
    """(start, end) epoch indices of the evaluated night: from lights off
    to the end of the human-scored main sleep period + margin (see the
    comment at SLEEP_PERIOD_MARGIN_MIN)."""
    sleep_idx = np.where(np.isin(y_true, ["N1", "N2", "N3", "REM"]))[0]
    sleep_idx = sleep_idx[sleep_idx >= lo_epoch]
    if len(sleep_idx) == 0:
        raise ValueError("human hypnogram contains no sleep after lights off")
    max_gap = MAIN_SLEEP_MAX_GAP_MIN * 60 // epoch_sec
    long_gaps = np.where(np.diff(sleep_idx) > max_gap)[0]
    main_end = sleep_idx[long_gaps[0]] if len(long_gaps) else sleep_idx[-1]
    end = min(main_end + SLEEP_PERIOD_MARGIN_MIN * 60 // epoch_sec + 1, len(y_true))
    return lo_epoch, end


def evaluate_one(psg_path: str, hyp_path: str, lights_off: datetime.time, subject_id: str = ""):
    """Run the full pipeline for a single recording: load, ground-truth
    hypnogram, YASA prediction, alignment and cropping to the night.
    Returns (y_true, y_pred) as two equal-length arrays of stage labels.
    Unscored epochs are kept as None (so durations stay correct for the
    clinical parameters) - drop them before computing agreement metrics."""
    log(f"  [{subject_id}] Loading EDF...")
    t0 = time.time()
    raw = load_recording(psg_path, hyp_path)
    log(f"  [{subject_id}] EDF loaded in {time.time()-t0:.1f}s. Channels: {raw.ch_names}")

    log(f"  [{subject_id}] Building ground-truth hypnogram (30s epochs)...")
    t0 = time.time()
    y_true = ground_truth_hypnogram(raw)
    log(f"  [{subject_id}] Ground truth done in {time.time()-t0:.1f}s ({len(y_true)} epochs)")

    log(f"  [{subject_id}] Creating YASA SleepStaging object...")
    t0 = time.time()
    sls = yasa.SleepStaging(
        raw,
        eeg_name=EEG_CHANNEL,
        eog_name=EOG_CHANNEL,
        emg_name=EMG_CHANNEL,
    )
    log(f"  [{subject_id}] SleepStaging created in {time.time()-t0:.1f}s")

    log(f"  [{subject_id}] Calling predict() (this is usually the slowest step)...")
    t0 = time.time()
    result = sls.predict()
    log(f"  [{subject_id}] predict() done in {time.time()-t0:.1f}s")
    # IMPORTANT: do not wrap this in np.asarray(result). YASA 0.7.0+
    # returns a yasa.Hypnogram object rather than a plain array, and
    # np.asarray() iterates it element-by-element, which rebuilds the
    # whole object on every single access (catastrophically slow - this
    # single line was the difference between "a few seconds" and
    # "several hours" during debugging). Use the documented .hypno
    # property instead, which returns the plain label array directly.
    y_pred = result.hypno.to_numpy()

    # YASA can occasionally return one epoch more or fewer than expected
    # due to rounding at the end of the recording - align lengths first.
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    # Crop both to the night (see the comment at SLEEP_PERIOD_MARGIN_MIN).
    # YASA still predicts on the full recording above, so its features
    # keep their full temporal context.
    start, end = analysis_window(y_true, lights_off_epoch(raw, lights_off))
    y_true, y_pred = y_true[start:end], y_pred[start:end]
    log(f"  [{subject_id}] Cropped to lights off -> main sleep end + {SLEEP_PERIOD_MARGIN_MIN} min: "
        f"epochs {start}-{end} ({(end - start) * EPOCH_SEC / 3600:.1f} h of {n * EPOCH_SEC / 3600:.1f} h)")
    return y_true, y_pred


def main():
    log("Starting main().")

    # Create the results folder if it doesn't already exist. exist_ok=True
    # means: if it's already there, don't raise an error, just continue.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log(f"Results will be saved to '{RESULTS_DIR}/'")

    psg_files = sorted(glob.glob(os.path.join(DATA_DIR, "*-PSG.edf")))
    log(f"Found {len(psg_files)} PSG files in '{DATA_DIR}'.")

    if not psg_files:
        log(f"No PSG files found - check the DATA_DIR path!")
        return

    # Lights-off clock time per (subject, night), used to crop each recording.
    subjects = pd.read_excel(SUBJECTS_XLS)
    lights_off_by_night = {(int(r.subject), int(r.night)): r.LightsOff for r in subjects.itertuples()}

    # Data funnel log (Article 10 / Annex IV data governance): every night
    # listed in SC-subjects.xls that does not end up in the results, with
    # the reason. Nights without a local PSG file never enter the loop
    # below, so they are logged here up front.
    excluded = []
    psg_nights = {(int(os.path.basename(p)[3:5]), int(os.path.basename(p)[5])) for p in psg_files}
    for (subj, night) in sorted(lights_off_by_night):
        if (subj, night) not in psg_nights:
            hyp = glob.glob(os.path.join(DATA_DIR, f"SC4{subj:02d}{night}*Hypnogram.edf"))
            excluded.append({"subject": f"SC4{subj:02d}{night}",
                             "reason": "PSG file not downloaded" + (" (hypnogram present)" if hyp else "")})

    results = []
    clinical_rows = []
    epoch_rows = []
    all_true, all_pred = [], []

    for i, psg_path in enumerate(psg_files, 1):
        subject_id = os.path.basename(psg_path).split("-")[0][:6]
        log(f"=== File {i}/{len(psg_files)}: {subject_id} ===")

        hyp_candidates = glob.glob(os.path.join(DATA_DIR, f"{subject_id}*Hypnogram.edf"))
        if not hyp_candidates:
            log(f"  [{subject_id}] No hypnogram found, skipping.")
            excluded.append({"subject": subject_id, "reason": "no hypnogram file"})
            continue
        hyp_path = hyp_candidates[0]

        lights_off = lights_off_by_night.get((int(subject_id[3:5]), int(subject_id[5])))
        if lights_off is None:
            log(f"  [{subject_id}] No LightsOff in {SUBJECTS_XLS}, skipping.")
            excluded.append({"subject": subject_id, "reason": f"no LightsOff in {SUBJECTS_XLS}"})
            continue

        try:
            y_true, y_pred = evaluate_one(psg_path, hyp_path, lights_off, subject_id)
        except Exception as e:
            log(f"  [{subject_id}] FAILED: {e}")
            excluded.append({"subject": subject_id, "reason": f"pipeline error: {e}"})
            continue

        # Per-epoch hypnograms of the evaluated window, for later checks
        # (bootstrap CIs, REM latency, worst-case inspection) without
        # re-running YASA. Unscored human epochs are stored as empty.
        epoch_rows.append(pd.DataFrame({"subject": subject_id, "epoch": np.arange(len(y_true)),
                                        "human": y_true, "yasa": y_pred}))

        # Agreement metrics only on epochs the human actually scored.
        mask = np.array([t is not None for t in y_true])
        y_true_scored, y_pred_scored = y_true[mask], y_pred[mask]

        acc = accuracy_score(y_true_scored, y_pred_scored)
        kappa = cohen_kappa_score(y_true_scored, y_pred_scored)
        f1_macro = f1_score(y_true_scored, y_pred_scored, average="macro", zero_division=0)

        log(f"  [{subject_id}] RESULT: accuracy={acc:.3f}  kappa={kappa:.3f}  macro-F1={f1_macro:.3f}  (n={mask.sum()} scored epochs)")

        results.append({
            "subject": subject_id,
            "accuracy": acc,
            "kappa": kappa,
            "macro_f1": f1_macro,
            "n_epochs": int(mask.sum()),
        })

        # Clinical parameters - human scoring vs YASA, for this subject.
        # Computed on the full window (unscored epochs kept as None), so
        # durations are not shortened by dropped epochs.
        stats_true = compute_sleep_stats(y_true)
        stats_pred = compute_sleep_stats(y_pred)
        clinical_rows.append({
            "subject": subject_id,
            "TST_true_min": stats_true["TST_min"], "TST_yasa_min": stats_pred["TST_min"],
            "SE_true_pct": stats_true["SE_pct"], "SE_yasa_pct": stats_pred["SE_pct"],
            "REM_lat_true_min": stats_true["REM_latency_min"], "REM_lat_yasa_min": stats_pred["REM_latency_min"],
            "WASO_true_min": stats_true["WASO_min"], "WASO_yasa_min": stats_pred["WASO_min"],
        })

        all_true.extend(y_true_scored)
        all_pred.extend(y_pred_scored)

    excluded_path = os.path.join(RESULTS_DIR, "excluded_recordings.csv")
    pd.DataFrame(excluded, columns=["subject", "reason"]).to_csv(excluded_path, index=False)
    log(f"Data funnel: {len(lights_off_by_night)} nights in {SUBJECTS_XLS}, "
        f"{len(results)} analysed, {len(excluded)} excluded -> {excluded_path}")

    if not results:
        log("No recordings were successfully processed.")
        return

    epochs_path = os.path.join(RESULTS_DIR, "yasa_sleepedfx_epochs.csv")
    pd.concat(epoch_rows).to_csv(epochs_path, index=False)
    log(f"Saved {epochs_path}")

    # --- Per-subject epoch-by-epoch metrics ---
    log("Saving per-subject results...")
    df = pd.DataFrame(results)
    results_path = os.path.join(RESULTS_DIR, "yasa_sleepedfx_pilot_results.csv")
    df.to_csv(results_path, index=False)
    log(f"Saved {results_path}")

    print("\n=== Summary across all recordings ===")
    print(df[["accuracy", "kappa", "macro_f1"]].describe())

    # --- Per-stage F1, pooled across all subjects ---
    # Useful to check whether YASA shows the same weakness pattern
    # (N1 lowest) reported in the original eLife validation paper.
    log("Computing per-stage F1 (pooled across all subjects)...")
    per_stage_f1 = f1_score(all_true, all_pred, labels=STAGES, average=None, zero_division=0)
    print("\n=== F1 per sleep stage (pooled across all subjects) ===")
    for stage, f1 in zip(STAGES, per_stage_f1):
        print(f"  {stage}: {f1:.3f}")

    # --- Clinical parameters: YASA vs human, per subject and averaged ---
    log("Computing and saving clinical parameters (TST/SE/REM latency/WASO)...")
    clin_df = pd.DataFrame(clinical_rows)
    clin_df["TST_diff_min"] = clin_df["TST_yasa_min"] - clin_df["TST_true_min"]
    clin_df["SE_diff_pct"] = clin_df["SE_yasa_pct"] - clin_df["SE_true_pct"]
    clin_df["REM_lat_diff_min"] = clin_df["REM_lat_yasa_min"] - clin_df["REM_lat_true_min"]
    clin_df["WASO_diff_min"] = clin_df["WASO_yasa_min"] - clin_df["WASO_true_min"]
    clin_df.to_csv(os.path.join(RESULTS_DIR, "yasa_sleepedfx_clinical_stats.csv"), index=False)
    log(f"Saved {os.path.join(RESULTS_DIR, 'yasa_sleepedfx_clinical_stats.csv')}")

    print("\n=== Clinical parameters - mean difference (YASA - human) ===")
    print(clin_df[["TST_diff_min", "SE_diff_pct", "REM_lat_diff_min", "WASO_diff_min"]].describe().loc[["mean", "std"]])

    log("main() finished.")


if __name__ == "__main__":
    main()
