# YASA Sleep Staging Audit - README

An audit of automatic sleep staging (YASA) against human scoring, on the public PhysioNet **Sleep-EDF Expanded (Sleep Cassette)** dataset: accuracy, clinical markers (TST, SE, WASO, REM latency) and the effect of age and sex on accuracy (GAMLSS).

Note on the sample: Sleep Cassette subjects are healthy Caucasian adults (aged 25-101), with no race/ethnicity data, so the results say nothing about bias across those groups.

## Folder structure

```
Projekat/
├── Code/                              (git repository - scripts only)
│   ├── README.md                      (this file)
│   ├── .gitignore
│   ├── environment.yml                (complete yasa_env conda environment)
│   ├── yasa_sleepedfx_pilot.py        (main script - run it first)
│   ├── gamlss_real_analysis.py        (GAMLSS: effect of age/sex on accuracy)
│   ├── bootstrap_ci.py                (95% CI, subject-level bootstrap)
│   ├── rem_latency_check.py           (REM latency and worst-recording check)
│   ├── distribution_check.py          (which distribution fits accuracy)
│   ├── reproducibility_info.py        (software/hardware versions + environment.yml)
│   └── generate_report_via_api.py     (optional - draft report via the Claude API)
└── Sleep Cassette/                    (data, outside git)
    ├── data/                          (EDF files from PhysioNet: *-PSG.edf, *-Hypnogram.edf)
    ├── supplementary/
    │   └── SC-subjects.xls            (age, sex, lights-off time per night)
    └── results/                       (all output CSV/TXT files)
```

The data and results live outside the git repository, so no EDF or results files end up in git.

## One-time installation

The easiest way is to create the environment directly from `environment.yml` (it includes R and `gamlss`). In the **Anaconda Prompt**, from the `Code` folder:

```
conda env create -f environment.yml
```

Manually, step by step:

```
conda create -n yasa_env python=3.11
conda activate yasa_env
pip install yasa mne scikit-learn pandas numpy matplotlib statsmodels xlrd scipy fitter
conda install -c conda-forge r-base r-gamlss rpy2
```

- Install R and `gamlss` DIRECTLY into `yasa_env` through conda-forge, not through a separate R installer from r-project.org.
- The "Solving environment" step for R can take 5-20 minutes; that is normal.
- `anthropic` is only needed for `generate_report_via_api.py` (`pip install anthropic`).

This is done **once**; you don't repeat it every time you sit down to work.

## Daily workflow (VS Code)

1. Open the `Code` folder in VS Code.
2. In the bottom right, check that the `yasa_env` interpreter is selected (if not: `Ctrl+Shift+P` → **Python: Select Interpreter** → `yasa_env`). The project already sets this through `.vscode/settings.json`.
3. Open a script and run it with the ▶ button (Run Python File).

The scripts use absolute paths to `C:/Users/karic/Documents/Mara/Projekat/Sleep Cassette/` (data, supplementary and results), so they can be run from any folder. If you move that folder, update the paths at the top of each script.

From a terminal (Anaconda Prompt):

```
conda activate yasa_env
python yasa_sleepedfx_pilot.py
```

In the VS Code terminal (PowerShell), `conda` is not recognised until conda is linked to PowerShell. Either call the env's Python directly:

```
& "C:\Users\karic\anaconda3\envs\yasa_env\python.exe" yasa_sleepedfx_pilot.py
```

or run once `& "C:\Users\karic\anaconda3\Scripts\conda.exe" init powershell` and open a new terminal. If the new terminal then reports that running scripts is disabled, also run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Script run order

1. **`yasa_sleepedfx_pilot.py`** - always first. Runs YASA on all recordings (about 25 min for 142 recordings) and writes the CSV files to `Sleep Cassette/results/`.
2. After #1, in any order:
   - **`gamlss_real_analysis.py`** - the main statistical analysis (age/sex → accuracy).
   - **`bootstrap_ci.py`** - confidence intervals.
   - **`rem_latency_check.py`** - REM latency and the worst recordings.
   - **`distribution_check.py`** - auxiliary: compares Normal, Beta and Weibull fits to accuracy.
   - **`generate_report_via_api.py`** - optional; requires `ANTHROPIC_API_KEY` as an environment variable.
3. **`reproducibility_info.py`** - last, to record the versions the results were produced with.

## Output files (`Sleep Cassette/results/`)

| File | Created by | Contents |
|---|---|---|
| `yasa_sleepedfx_pilot_results.csv` | pilot | accuracy, kappa, macro-F1 per recording |
| `yasa_sleepedfx_clinical_stats.csv` | pilot | TST, SE, REM latency, WASO - human vs YASA, per recording |
| `yasa_sleepedfx_epochs.csv` | pilot | per-epoch stage labels (human and YASA) |
| `excluded_recordings.csv` | pilot | excluded nights and the reason |
| `gamlss_per_subject.csv` | gamlss | age, sex, accuracy and model prediction per subject (for plotting) |
| `gamlss_age_curve.csv` | gamlss | model curve + 5th-95th centile band by age and sex (for plotting) |
| `bootstrap_ci.csv` | bootstrap_ci | means and 95% CIs |
| `rem_latency_check.txt` | rem_latency_check | REM latency and worst-recording analysis |
| `reproducibility_info.txt` | reproducibility_info | software versions, R/gamlss, OS, hardware |

## Methodology in brief

- **Analysis window:** Sleep Cassette recordings last about 23 h. Only the night is analysed: from lights off (`LightsOff` from `SC-subjects.xls`) to the end of the human-scored main sleep period + 30 min. The main sleep period ends at the first wake gap longer than 2 h.
- **Unscored epochs** (`Sleep stage ?`, `Movement time`) are dropped only from accuracy/kappa/F1, not from TST/WASO/SE.
- **Repeated measures:** the 142 recordings come from 78 subjects (64 with two nights). GAMLSS uses one row per subject (the night average), and the bootstrap CI resamples subjects, not recordings.

## Known pitfalls (so you don't lose time on the same problems again)

- **OpenMP crash/hang** - `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` must be the FIRST line of code, before any `import` statements. It is already built into all scripts.
- **`np.asarray(sls.predict())` is SLOW** - YASA 0.7.0+ returns a special `Hypnogram` object, not a plain array. Always use `.hypno.to_numpy()` - the difference was seconds versus HOURS.
- **Stage labels** - YASA 0.7.0+ uses `WAKE` and `REM`, not `W` and `R`.
- **Windows paths** - use `r"C:\path"` (raw string) or `"C:/path"`, never a plain `"C:\path"` (e.g. `\r` becomes a carriage-return character).
- **Wrong Python** - if you get `ModuleNotFoundError: No module named 'mne'`, the script was run by a different Python (e.g. Inkscape's), not the one in `yasa_env`. Select `yasa_env` as the interpreter.
- **R "LoadLibrary failure" / `grDevices.dll`** - happens when Python is started without `conda activate` (e.g. ▶ in VS Code). That is why `gamlss_real_analysis.py` and `reproducibility_info.py` set up `PATH` themselves at the start. If you write a new script that uses rpy2, copy that block.
- **scikit-learn warning** (`InconsistentVersionWarning`) - YASA's model was trained with scikit-learn 0.24, and the environment has a newer version. The results are stable, but this is listed as a limitation.

## Removing the environment

```
conda env remove -n yasa_env
```

This removes the environment with all its packages. The scripts and results stay untouched in the `Code` and `Sleep Cassette` folders - you only delete the "kitchen", not the "recipes and the food". Recreate the environment with `conda env create -f environment.yml`.
