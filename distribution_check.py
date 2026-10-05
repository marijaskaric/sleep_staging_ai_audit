"""
Which distribution fits YASA accuracy?
======================================
Histogram of the per-recording accuracy values, with fitted Normal and
Beta curves overlaid, so you can SEE which theoretical distribution
matches the shape of the data - not just read p-values about it. This
informs the distribution family used in gamlss_real_analysis.py.

Also includes:
  - an automated ranking of candidate distributions (via the optional
    `fitter` package)
  - a fair AIC comparison of a properly constrained Beta vs a Weibull

Requires Sleep Cassette/results/yasa_sleepedfx_pilot_results.csv (from
yasa_sleepedfx_pilot.py); run from the project root. Saves the plot as
accuracy_distribution_comparison.png in the project root.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoids the OpenMP duplicate-library crash

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, norm

# ---- 1. Load the per-recording accuracy values ---------------------------

df = pd.read_csv("C:/Users/karic/Documents/Mara/Projekat/Sleep Cassette/results/yasa_sleepedfx_pilot_results.csv")
data = df["accuracy"].values

# ---- 2. Fit a Normal distribution to the data ------------------------------
# .fit() finds the mean and std dev that best match your data (maximum
# likelihood estimation) - this is what OLS silently assumes is true.
mu, sigma = norm.fit(data)
print(f"Fitted Normal: mean={mu:.3f}, std={sigma:.3f}")

# ---- 3. Fit a Beta distribution to the data --------------------------------
# Beta needs values strictly between 0 and 1, so clip the (theoretically
# possible) exact 0 or 1 slightly inward - same as gamlss_real_analysis.py.
epsilon = 1e-4
data_beta = np.clip(data, epsilon, 1 - epsilon)
a, b, loc, scale = beta.fit(data_beta, floc=0, fscale=1)
print(f"Fitted Beta: a={a:.2f}, b={b:.2f}")

# ---- 4. Plot histogram with both curves overlaid ---------------------------

x = np.linspace(0, 1, 200)

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(data, bins=15, density=True, alpha=0.5, color="gray", label="Actual data (histogram)")
ax.plot(x, norm.pdf(x, mu, sigma), color="crimson", linewidth=2, label="Fitted Normal curve")
ax.plot(x, beta.pdf(x, a, b), color="steelblue", linewidth=2, label="Fitted Beta curve")
ax.set_xlabel("Accuracy")
ax.set_ylabel("Density")
ax.set_title("Which curve actually follows the shape of the real data?")
ax.legend()
fig.tight_layout()
fig.savefig("accuracy_distribution_comparison.png", dpi=120)
plt.show()
print("\nSaved plot as accuracy_distribution_comparison.png")

# ---- 5. Let the computer try MANY distributions and rank them -------------
# Instead of guessing by eye, `fitter` fits a whole list of candidate
# distributions and sorts them by goodness-of-fit (sum of squared error
# between the histogram and each fitted curve, by default).
try:
    from fitter import Fitter

    f = Fitter(data, distributions=["norm", "beta", "lognorm", "gamma", "weibull_min"])
    f.fit()
    print("\n=== Automated ranking of candidate distributions ===")
    print(f.summary())
except ImportError:
    print("\n(Install 'fitter' with: pip install fitter, to also get an "
          "automated ranking of candidate distributions.)")
# ---- 6. FAIR comparison: properly-constrained Beta vs free Weibull --------
# fitter's own "beta" row used an UNCONSTRAINED fit (it tried to guess the
# bounds itself, and did so badly - hence AIC=inf). That is not a fair test
# of whether Beta suits accuracy data; it only shows that letting Beta's
# bounds float freely is a bad idea (which we already knew - accuracy is
# bounded at 0 and 1 by definition, so we should fix loc/scale, not fit them).
# Here we compare our PROPERLY constrained Beta against Weibull on equal
# footing, using AIC computed directly from each model's log-likelihood.
from scipy.stats import weibull_min

ll_beta = np.sum(beta.logpdf(data_beta, a, b, loc=0, scale=1))
aic_beta_fair = 2 * 2 - 2 * ll_beta  # Beta here has 2 free parameters (a, b)

c, loc_w, scale_w = weibull_min.fit(data)
ll_weibull = np.sum(weibull_min.logpdf(data, c, loc_w, scale_w))
aic_weibull = 2 * 3 - 2 * ll_weibull  # Weibull here has 3 free parameters

print("\n=== Fair comparison: constrained Beta vs free Weibull ===")
print(f"Beta AIC:    {aic_beta_fair:.2f}")
print(f"Weibull AIC: {aic_weibull:.2f}")
print(f"Difference (Weibull - Beta): {aic_weibull - aic_beta_fair:.2f}")
print(
    "\nRule of thumb for reading the AIC difference:\n"
    "  < 2   -> essentially no difference, either model is fine\n"
    "  4-7   -> considerably less support for the higher-AIC model\n"
    "  > 10  -> essentially no support for the higher-AIC model\n"
    "Note Beta also has the advantage of a built-in, principled reason "
    "to be bounded at (0,1) - Weibull only stays in that range by "
    "coincidence of its fitted parameters, not by construction."
)
