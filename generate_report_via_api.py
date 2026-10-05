"""
Generate a draft report via the Claude API from the YASA pilot results
======================================================================
Follow-up to yasa_sleepedfx_pilot.py - expects
Sleep Cassette/results/yasa_sleepedfx_pilot_results.csv to exist.

What this script does:
1) Loads the computed metrics (accuracy, kappa, macro-F1 - numbers only,
   never raw PSG data; see the privacy note below).
2) Builds a prompt from AGGREGATE statistics of those numbers.
3) Sends it to the Anthropic API and asks for a short findings section
   (in Serbian).
4) Saves the answer as pilot_izvestaj.md.

------------------------------------------------------------------------
SETUP
------------------------------------------------------------------------
1) pip install anthropic pandas

2) Create an API key at https://console.anthropic.com (Settings -> API Keys)

3) NEVER write the key into the code. Set it as an environment variable
   before running:

   Linux/Mac (in the terminal, before running the script):
       export ANTHROPIC_API_KEY="your-key-here"

   Windows (PowerShell):
       $env:ANTHROPIC_API_KEY="your-key-here"

------------------------------------------------------------------------
PRIVACY NOTE
------------------------------------------------------------------------
This script sends ONLY aggregate numbers (mean accuracy, quartiles, a
count of outliers) - never raw EEG signals, per-recording rows, recording
or subject IDs, or anything from the PSG files themselves. This follows
the data-minimisation principle: send only what is actually needed.
Keep it that way if the script is reused on other datasets - aggregates
per subgroup (e.g. "age group 25-40, N=30, mean accuracy 0.74") are fine,
individual identifiers are not.
------------------------------------------------------------------------
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoids the OpenMP duplicate-library crash

import anthropic
import pandas as pd

# ---- 1. Load the results from the previous step -------------------------

df = pd.read_csv("C:/Users/karic/Documents/Mara/Projekat/Sleep Cassette/results/yasa_sleepedfx_pilot_results.csv")

summary_stats = df[["accuracy", "kappa", "macro_f1"]].describe()

# ---- 2. Build the text summary of the numbers (this goes into the prompt)

# A plain format string, no fancy library - easy to check exactly what is
# sent before it is sent. AGGREGATES ONLY: no per-recording rows and no
# recording IDs. Outliers are sent only as a count of recordings below a
# threshold.
low_threshold = df["accuracy"].mean() - 2 * df["accuracy"].std()
n_low = int((df["accuracy"] < low_threshold).sum())
metrics_text = f"""
Broj obrađenih snimaka: {len(df)}

Sažetak metrika preko svih snimaka:
{summary_stats.to_string()}

Broj snimaka sa accuracy ispod (prosek - 2 SD = {low_threshold:.3f}): {n_low}
"""

print("=== THIS IS WHAT WILL BE SENT TO THE API (check before sending) ===")
print(metrics_text)
print("===================================================================")

# ---- 3. Create the client and send the request --------------------------

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"]  # read from the environment variable
)

# The prompt is in Serbian because the report itself is written in Serbian.
prompt = f"""Ti si asistent koji pomaže u pisanju tehničkog izveštaja o
validaciji algoritma za automatsko stažiranje sna (YASA), testiranog na
Sleep-EDF Expanded (Sleep Cassette) skupu podataka.

Evo izračunatih metrika:
{metrics_text}

Napiši kratak nalaz (najviše 300 reči) koji:
1. Sumira ukupnu performansu (accuracy, kappa, macro-F1) u odnosu na
   poznate referentne vrednosti iz literature (YASA eLife rad 2021:
   medijalna tačnost ~87.5% na NSRR podacima, ~86.6% na zdravim
   odraslima iz Dreem Open Dataset).
2. Napomene ako postoje snimci sa izrazito lošijim rezultatom od
   proseka (dat je samo njihov broj), i da to treba dalje istražiti
   (ne spekuliši o uzroku bez podataka).
3. Zaključi jednom rečenicom da li je pilot spreman za sledeći korak
   analize.

Piši na srpskom jeziku, faktički i suzdržano - ovo je nacrt za tehnički
izveštaj, ne marketinški tekst."""

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1000,
    messages=[{"role": "user", "content": prompt}],
)

report_text = response.content[0].text

# ---- 4. Save the report -------------------------------------------------

with open("pilot_izvestaj.md", "w", encoding="utf-8") as f:
    f.write("# Nalaz - YASA pilot na Sleep-EDF Expanded\n\n")
    f.write(report_text)

print("\nReport saved to pilot_izvestaj.md")
print("\n--- Report content ---\n")
print(report_text)
