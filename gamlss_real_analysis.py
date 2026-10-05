"""
GAMLSS age/sex analysis of YASA accuracy, via rpy2
==================================================
The real model behind the age_glm_practice.py stand-in. This script
calls R's `gamlss` package directly from Python via rpy2 - no need to
ever open a separate R terminal.

What this does differently from the plain OLS regression:
  - one row per SUBJECT (the two nights are averaged), so the
    observations are independent
  - models accuracy with a BETA distribution (bounded 0-1), not a
    normal distribution - this addresses the skew/kurtosis problem seen
    in the OLS diagnostics
  - uses a smooth, non-linear spline term for age (via pb(), a
    penalized B-spline - gamlss picks how wiggly the curve should be
    automatically, rather than us guessing)
  - tests each term with a likelihood-ratio test (drop1)
  - this is the same modeling family used in the "Beyond accuracy" and
    SLEEPYLAND papers
A Weibull model with the same formula is fitted for comparison.

Requires (run from the project root):
  Sleep Cassette/results/yasa_sleepedfx_pilot_results.csv  (from yasa_sleepedfx_pilot.py)
  Sleep Cassette/supplementary/SC-subjects.xls             (age/sex, from PhysioNet)
  R + gamlss + rpy2 installed in the yasa_env conda environment.

Outputs (for plotting), in Sleep Cassette/results/:
  gamlss_per_subject.csv  observed and fitted accuracy per subject
  gamlss_age_curve.csv    fitted curve + 5th-95th centile band per age and sex
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoids the OpenMP duplicate-library crash

import pandas as pd

# rpy2 needs to know where R lives inside the conda env.
ENV_DIR = r"C:\Users\karic\anaconda3\envs\yasa_env"
os.environ["R_HOME"] = os.path.join(ENV_DIR, "lib", "R")
# When python.exe is started without `conda activate` (e.g. VS Code's Run
# button), the env's DLL folders are missing from PATH and R picks up
# incompatible DLLs ("LoadLibrary failure"). Put them first on PATH.
os.environ["PATH"] = os.pathsep.join([
    os.path.join(ENV_DIR, "lib", "R", "bin", "x64"),
    os.path.join(ENV_DIR, "Library", "mingw-w64", "bin"),
    os.path.join(ENV_DIR, "Library", "usr", "bin"),
    os.path.join(ENV_DIR, "Library", "bin"),
    ENV_DIR,
    os.environ.get("PATH", ""),
])
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

# ---- 1. Load and merge data (same logic as age_glm_practice.py) ---------

results = pd.read_csv("C:/Users/karic/Documents/Mara/Projekat/Sleep Cassette/results/yasa_sleepedfx_pilot_results.csv")
demographics = pd.read_excel("C:/Users/karic/Documents/Mara/Projekat/Sleep Cassette/supplementary/SC-subjects.xls").rename(columns={"sex (F=1)": "sex"})  # 1 = female, 2 = male

results["subject_num"] = results["subject"].str[3:5].astype(int)
results["night"] = results["subject"].str[5].astype(int)

merged = results.merge(
    demographics[["subject", "night", "age", "sex"]],
    left_on=["subject_num", "night"],
    right_on=["subject", "night"],
    suffixes=("", "_demo"),
)

print(f"Merged dataset: {len(merged)} rows")
print(merged[["subject", "age", "sex", "accuracy"]].describe())

# gamlss's beta family (BE) requires the outcome to be STRICTLY between
# 0 and 1 (not equal to 0 or 1). accuracy_score can in theory return
# exactly 1.0 for a perfect subject - nudge any boundary values slightly
# inward so the model doesn't reject them. This is a standard, documented
# trick for beta regression, not a data manipulation - it moves values by
# a fraction of a percent at most.
epsilon = 1e-4
merged["accuracy_beta"] = merged["accuracy"].clip(lower=epsilon, upper=1 - epsilon)

# sex as a categorical factor (not a raw 1/2 number) - gamlss should
# treat it as a group label, not as if "2" were twice "1".
merged["sex"] = merged["sex"].astype(str)

# The recordings come from fewer subjects - most were recorded on two
# nights, and two nights of the same person are not independent
# observations. Treating them as independent makes p-values too
# optimistic. Age and sex are subject-level (same on both nights), so we
# average accuracy per subject and fit one row per subject. (A random
# intercept, random()/re(), was tried instead: in gamlss it absorbs the
# subject-level age effect and the age test becomes unusable.)
n_recordings = len(merged)
per_subject = (
    merged.groupby("subject_num")
    .agg(accuracy=("accuracy", "mean"), age=("age", "mean"), sex=("sex", "first"),
         n_nights=("accuracy", "size"))
    .reset_index()
)
per_subject["accuracy_beta"] = per_subject["accuracy"].clip(lower=epsilon, upper=1 - epsilon)
print(f"\n{n_recordings} recordings -> {len(per_subject)} subjects "
      f"({(per_subject['n_nights'] == 2).sum()} with two nights averaged)")

# ---- 2. Hand the data to R -------------------------------------------
# Named `dat`, not `df`: df is also a base R function, and some gamlss
# internals then pick up the function instead of the data.

pandas2ri.activate()
ro.globalenv["dat"] = pandas2ri.py2rpy(per_subject[["accuracy_beta", "age", "sex"]])
ro.r("dat$sex <- factor(dat$sex)")

# ---- 3. Fit the GAMLSS model in R --------------------------------------

gamlss = importr("gamlss")

# BE() = beta distribution family, bounded on (0,1) - appropriate for
# accuracy scores. pb(age) = a penalized B-spline smooth term for age,
# letting the age effect be non-linear instead of forcing a straight
# line (this is the key difference from the plain OLS model earlier).
ro.r("""
ctl <- gamlss.control(trace = FALSE)
model <- gamlss(accuracy_beta ~ pb(age) + sex, family = BE(), data = dat, control = ctl)
""")

print("\n=== GAMLSS model summary (Beta) ===")
print(ro.r("summary(model)"))

# ---- 4. Test each term with a likelihood-ratio test ---------------------
# summary() itself warns that its standard errors are not accurate when
# the model has smooth terms. drop1() refits the model without each term
# and compares the fits (LRT) - the more reliable test of "does age matter".
print("\n=== Likelihood-ratio test per term (drop1) ===")
print(ro.r("drop1(model, what = 'mu')"))

# ---- 5. Check whether the age term is actually non-linear ---------------
# Effective degrees of freedom of the spline (the count includes the
# linear part): about 2 means the fitted curve is essentially a straight
# line; clearly above 2 means real curvature was found.
print("\n=== Effective degrees of freedom of the age spline ===")
print(ro.r("edfAll(model)"))

# ---- 6. Get the model's ACTUAL predictions at specific ages -------------
# predict() asks the model itself for its spline-corrected prediction.
ro.r("""
newdata <- data.frame(age = c(30, 90), sex = factor(c("1", "1"), levels = levels(dat$sex)))
predictions <- predict(model, newdata = newdata, type = "response", data = dat)
""")

print("\n=== Model predictions (spline-corrected) ===")
print("Age 30, sex=1 (F):", ro.r("predictions[1]")[0])
print("Age 90, sex=1 (F):", ro.r("predictions[2]")[0])

# ---- 6b. Export data for plotting ----------------------------------------
# (a) one row per subject: observed (night-averaged) accuracy, age, sex and
#     the Beta model's fitted mean for that subject;
# (b) the fitted curve on an age grid, per sex, with the 5th-95th centile
#     band of the Beta distribution (where 90% of subjects are expected).
per_subject["sex_label"] = per_subject["sex"].map({"1": "F", "2": "M"})
per_subject["predicted_accuracy"] = list(ro.r("fitted(model, 'mu')"))
subjects_csv = os.path.join("C:/Users/karic/Documents/Mara/Projekat/Sleep Cassette", "results", "gamlss_per_subject.csv")
per_subject[["subject_num", "age", "sex_label", "n_nights", "accuracy", "predicted_accuracy"]].to_csv(
    subjects_csv, index=False)

ro.r("""
grid <- expand.grid(age = seq(min(dat$age), max(dat$age), by = 1), sex = levels(dat$sex))
pa <- predictAll(model, newdata = grid, data = dat, type = "response")
curve <- data.frame(age = grid$age, sex = grid$sex, predicted_accuracy = pa$mu,
                    centile_05 = qBE(0.05, mu = pa$mu, sigma = pa$sigma),
                    centile_95 = qBE(0.95, mu = pa$mu, sigma = pa$sigma))
""")
curve = ro.r("curve")  # pandas2ri.activate() already converts it to a DataFrame
curve["sex"] = curve["sex"].astype(str).map({"1": "F", "2": "M"})
curve_csv = os.path.join("C:/Users/karic/Documents/Mara/Projekat/Sleep Cassette", "results", "gamlss_age_curve.csv")
curve.to_csv(curve_csv, index=False)
print(f"\nSaved {subjects_csv} ({len(per_subject)} subjects) and {curve_csv} ({len(curve)} rows)")

# ---- 7. Model 2 - same formula, Weibull family instead of Beta -----------
# Weibull showed a better raw AIC on the unconditional distribution check
# (distribution_check.py), so it's worth comparing here too, on the actual
# regression model (not just the marginal shape of accuracy).
ro.r("""
model2 <- gamlss(accuracy_beta ~ pb(age) + sex, family = WEI(), data = dat, control = ctl)
""")

print("\n=== Model 2 (Weibull family) summary ===")
print(ro.r("summary(model2)"))

print("\n=== Model comparison: Beta vs Weibull (lower AIC/GAIC = better fit) ===")
print(ro.r("GAIC(model, model2)"))

# Weibull is not bounded at 1, so a better AIC does not guarantee sensible
# predictions. Report how much predicted probability mass falls above 100%
# accuracy (averaged over subjects) - it should be negligible.
mass_above_1 = ro.r("mean(1 - pWEI(1, mu = fitted(model2, 'mu'), sigma = fitted(model2, 'sigma')))")[0]
print(f"\nWeibull model: mean predicted probability of accuracy > 1 = {mass_above_1:.4f}")
print("(If this is not close to 0, Weibull is not a valid model for accuracy "
      "despite its AIC - prefer Beta, which is bounded on (0, 1) by construction.)")
