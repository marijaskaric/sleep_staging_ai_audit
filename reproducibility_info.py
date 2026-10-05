"""
Reproducibility snapshot
========================
Records the software and hardware the results were produced with:
Python package versions, R + gamlss versions (through rpy2), OS and CPU,
and the git commit of the scripts (if any). Also exports the full conda
environment to environment.yml in the project root.

Writes Sleep Cassette/results/reproducibility_info.txt.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import datetime
import platform
import subprocess
import sys
from importlib.metadata import version, PackageNotFoundError

ENV_DIR = r"C:\Users\karic\anaconda3\envs\yasa_env"
RESULTS_DIR = "C:/Users/karic/Documents/Mara/Projekat/Sleep Cassette/results"
PACKAGES = ["yasa", "mne", "scikit-learn", "lightgbm", "numpy", "scipy", "pandas",
            "statsmodels", "matplotlib", "xlrd", "rpy2", "fitter", "anthropic"]

lines = [f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
         f"Python: {sys.version.split()[0]} ({sys.executable})",
         f"OS: {platform.platform()}",
         f"CPU: {platform.processor()} ({os.cpu_count()} logical cores)", ""]

lines.append("Python packages:")
for pkg in PACKAGES:
    try:
        lines.append(f"  {pkg}=={version(pkg)}")
    except PackageNotFoundError:
        lines.append(f"  {pkg}: not installed")

# R and gamlss, through rpy2 (same PATH setup as gamlss_real_analysis.py).
os.environ["R_HOME"] = os.path.join(ENV_DIR, "lib", "R")
os.environ["PATH"] = os.pathsep.join([
    os.path.join(ENV_DIR, "lib", "R", "bin", "x64"),
    os.path.join(ENV_DIR, "Library", "mingw-w64", "bin"),
    os.path.join(ENV_DIR, "Library", "usr", "bin"),
    os.path.join(ENV_DIR, "Library", "bin"),
    ENV_DIR,
    os.environ.get("PATH", ""),
])
try:
    import rpy2.robjects as ro
    lines += ["", "R: " + ro.r("R.version.string")[0]]
    for r_pkg in ["gamlss", "gamlss.dist"]:
        lines.append(f"  {r_pkg}==" + ro.r(f"as.character(packageVersion('{r_pkg}'))")[0])
except Exception as e:
    lines += ["", f"R: could not query ({e})"]

try:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    lines += ["", f"Git commit: {commit}" + (" (uncommitted changes present)" if dirty else "")]
except (subprocess.CalledProcessError, FileNotFoundError):
    lines += ["", "Git commit: none (repository has no commits yet)"]

lines += ["", "Random seeds: bootstrap_ci.py uses SEED = 20261002. YASA prediction and the",
          "gamlss RS fitting algorithm are deterministic (no seed involved)."]

text = "\n".join(lines) + "\n"
print(text)
with open(os.path.join(RESULTS_DIR, "reproducibility_info.txt"), "w", encoding="utf-8") as f:
    f.write(text)

# conda is often not on PATH outside an activated Anaconda prompt (e.g. VS
# Code's Run button), so fall back to the base install's conda.exe.
conda_exe = os.environ.get("CONDA_EXE") or os.path.join(os.path.dirname(os.path.dirname(ENV_DIR)), "Scripts", "conda.exe")
env_yml = subprocess.run([conda_exe, "env", "export", "-p", ENV_DIR, "--no-builds"],
                         capture_output=True, text=True)
if env_yml.returncode == 0:
    # Drop the machine-specific "prefix:" line (local path incl. user name);
    # `conda env create -f environment.yml` does not need it.
    lines_yml = [ln for ln in env_yml.stdout.splitlines() if not ln.startswith("prefix:")]
    with open("environment.yml", "w", encoding="utf-8") as f:
        f.write("\n".join(lines_yml) + "\n")
    print("Saved environment.yml")
else:
    print("conda env export failed:", env_yml.stderr.strip())
