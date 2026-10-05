# Audit Report: YASA on Sleep-EDF Expanded

**Performance and bias analysis** (practice exercise in applying the methodology)

## Document Control

| Field | Value |
| --- | --- |
| Document version | 1.0 (draft) |
| Date | 5 October 2026 |
| Author | MSK |
| Status | Practice exercise in applying the methodology; not a regulatory submission |
| Repository | commit 8bd916223e083001b4c5e7193b84f1d84e43660b, link https://github.com/marijaskaric/sleep\_staging\_ai\_audit |
| Review and approval | Not performed |

### Document Revision History

| Version | Date | Description |
| --- | --- | --- |
| 1.0 | 5 Oct 2026 | First issue in audit report format (draft) |

## 1. Summary

This exercise applies the conformity-assessment approach behind Articles 10 and 15 of the EU AI Act to YASA, a publicly available open-source model for automatic sleep scoring (Vallat & Walker, 2021). The model was tested on Sleep-EDF Expanded (Kemp et al., 2000; Goldberger et al., 2000), a dataset it was not trained on: 142 nights from 78 subjects aged 25 to 101.

Mean accuracy per recording was 69.2% (95% CI 67.4-71.1%), with kappa 0.569 (95% CI 0.545-0.594) and macro-F1 0.584 (95% CI 0.563-0.606). N1 was the weakest stage (F1 0.281). A GAMLSS model on 78 independent subjects shows accuracy falling with age (LRT p=0.00004; 74.7% at age 30 versus 64.7% at age 90) and no significant effect of sex (p=0.22). YASA overestimated WASO by 30.9 min on average and underestimated REM latency by 68.8 min; in 88 of 142 recordings it placed the first REM within the first 5 minutes of sleep, and the cause is still unknown.

Race/ethnicity and clinical indices (AHI, PLMI) could not be examined because the dataset does not contain them. No acceptance criteria were set in advance, so the report describes how the model behaves and does not declare a pass or fail. It is an educational exercise, not a regulatory submission.

## 2. Purpose, Scope and Regulatory Framework

### 2.1 Purpose and scope

This document is a **practice exercise** in applying the conformity-assessment approach that the EU AI Act (European Parliament & Council, 2024) and the MDR (European Parliament & Council, 2017) expect for AI components in medical devices. It was not prepared for, and has not been sent to, any regulator or notified body. **YASA and Sleep-EDF Expanded are not a CE-marked product or its validation dossier.** They are a public open-source model and a public research dataset, used here to show what such an audit could look like.

### 2.2 Regulatory framework and applicability

The structure of the report follows:

- **Article 10** (Regulation (EU) 2024/1689): data governance, examination of representativeness and discriminatory impact
- **Article 15**: accuracy and robustness
- **Annex IV, point 2(g)**: metrics for accuracy, robustness and discriminatory impact
- The Johner Institute AI Guideline (n.d.) as a practical checklist that maps the above articles to concrete sections of the report

**Timing.** The Digital Omnibus (European Parliament & Council, 2026), in force since 27 July 2026, postponed the AI Act's requirements for AI in medical devices from August 2027 to 2 August 2028. It also made other targeted changes, for example broadening the legal basis for processing special categories of data to detect bias.

**Legal applicability.** Whether the AI Act would apply to YASA depends on whether the final product would be medical device software that needs a notified body under the MDR; Article 6(1) ties high-risk status to exactly that condition. Software with a declared diagnostic or monitoring purpose typically falls under MDR Rule 11 (Annex VIII), usually Class IIa or higher, so a notified body would be needed. If the work is purely research with no plan to place anything on the market, **Article 2(8) of the AI Act excludes R&D activity before placing on the market** from most of its obligations.

Because it is not settled whether a real product would go toward CE marking, this report applies the full AI Act/MDR framework anyway. That is the more conservative approach and the more useful one for an exercise. It does not claim the framework is legally binding for this audit.

### 2.3 Statement on data source and purpose

The report uses only the publicly available Sleep-EDF Expanded dataset (PhysioNet). The analysis was done for educational purposes only, not for commercial use. It contains no confidential information about individual subjects: only aggregate statistics from a public, already de-identified research dataset are used, and no recording or subject ID appears in this document. The script that drafts the report text (generate\_report\_via\_api.py) sends only aggregates, never recording IDs, in line with data minimization.

## 3. Methodology

### 3.1 Model

YASA v0.7.0 (Vallat & Walker, 2021), run without age/sex metadata as input to the classifier (an estimate of baseline, "blind" performance). YASA uses LightGBM (Ke et al., 2017) and MNE-Python (Gramfort et al., 2013).

### 3.2 Dataset and data flow

Sleep-EDF Expanded, Sleep Cassette subset (Kemp et al., 2000; Goldberger et al., 2000).

| Step | Number of recordings | Note |
| --- | --- | --- |
| Sleep-EDF Expanded, Sleep Cassette subset (publicly available) | 153 (78 subjects) | Full publicly available subset |
| Included in the analysis | 142 | 11 omitted: PSG file not downloaded (hypnogram exists) |
| Independent subjects (after averaging 2 nights) | 78 (64 with two nights) | Used for GAMLSS |

All 11 omitted nights were omitted for the same reason: the PSG file was not downloaded, although the hypnogram exists. No recording was excluded during the analysis itself. Each of those 11 subjects has their other night in the analysis, so the number of subjects is still 78. The reason for each omitted recording is written to excluded\_recordings.csv.

### 3.3 Channels and reference hypnograms

- **Channels:** EEG Fpz-Cz, EOG horizontal, EMG submental. In Sleep Cassette recordings the submental EMG is an envelope sampled at 1 Hz, not a raw signal.
- **Reference:** hypnograms scored by human experts according to Rechtschaffen and Kales (1968), based on Fpz-Cz/Pz-Oz EEG, available in the dataset itself.
- **Epochs:** 30 s.

### 3.4 Analysis window

The window starts at LightsOff from SC-subjects.xls (the same moment for the human and for YASA). It ends at the end of the human-scored main sleep period plus 30 min; the main sleep period ends at the first wake gap longer than 2 h. Unscored epochs are not dropped before computing TST/WASO/SE; they are dropped only when computing accuracy, kappa and F1.

Why the window is restricted: Sleep Cassette recordings last about 23 h (day, night and the following day). In the first version (v0), about 16 h of daytime wake artificially raised accuracy, and a few spurious YASA sleep epochs during the day stretched YASA's sleep period over the whole recording, so daytime wake was counted as WASO (mean difference +482 min). The history of changes is in Appendix A.

### 3.5 Metrics and statistical methods

- **Metrics:** accuracy, Cohen's kappa (Cohen, 1960), macro-F1, computed per recording, then aggregated across all 142 recordings.
- **Confidence intervals:** subject-level bootstrap (Efron, 1979), 10,000 resamples of subjects, each with both of their nights (seed 20261002, script bootstrap\_ci.py).
- **Effect of age and sex:** GAMLSS (Rigby & Stasinopoulos, 2005), Beta and Weibull families, on 78 independent subjects (two nights averaged to avoid pseudo-replication). Each factor was tested with a likelihood-ratio test (drop1). In the model only the mean (mu) depends on age and sex, while the dispersion (sigma) has only an intercept, i.e. it is assumed constant. This is therefore not a full distributional analysis as in Bechny et al. (2025), which also models dispersion.

### 3.6 Acceptance criteria and analysis plan

No acceptance criteria (for example a minimum accuracy) were defined in advance, and the analysis plan was not written down before the analysis was run. The methodology was refined over three iterations after the results had been seen (Appendix A). The results therefore describe how the model behaves; they are not a pass/fail test.

## 4. Results

### 4.1 Demographics

The dataset covers subjects from 25 to 101 years of age. Sex is coded as 1 = female, 2 = male (column "sex (F=1)" in SC-subjects.xls), so a mean of 1.46 means that 54% of recordings are from women.

| Variable | Mean | SD | Min | Q25 | Median | Q75 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Age (years) | 58.8 | 22.0 | 25 | 38 | 57 | 73 | 101 |
| Sex (1 = F, 2 = M) | 1.46 | 0.50 | 1 | 1 | 1 | 2 | 2 |

### 4.2 Overall performance

Performance was computed per recording (not pooled across all epochs), across all 142 recordings.

| Metric | Mean | SD | Min | Q25 | Median | Q75 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Accuracy | 0.692 | 0.099 | 0.372 | 0.655 | 0.709 | 0.760 | 0.862 |
| Cohen's kappa | 0.569 | 0.131 | 0.092 | 0.507 | 0.585 | 0.656 | 0.810 |
| Macro-F1 | 0.584 | 0.111 | 0.235 | 0.515 | 0.599 | 0.661 | 0.779 |

**Confidence intervals:** the 142 recordings come from 78 subjects (64 with two nights), so a normal approximation with n=142 treats two nights of the same person as independent and gives an interval that is too narrow. The subject-level bootstrap gives a 95% CI for mean accuracy of \[0.674, 0.711\], for kappa \[0.545, 0.594\] and for macro-F1 \[0.563, 0.606\]. For comparison, a naive normal approximation with n=142 would give a narrower interval (\[0.676, 0.708\] for accuracy).

### 4.3 Performance by sleep stage

&#91;embedded content: computed from predictions and hypnograms, pooled across all 142 recordings\]

N1 is the weakest stage (F1 = 0.281). Wake is 0.735, N2 0.769, N3 0.665 and REM 0.698. F1 per stage was computed pooled across all recordings, so it differs from the per-recording macro-F1 in the table above.

### 4.4 Effect of age and sex

| Model (78 subjects) | Predictor | Coefficient | Test | p-value | Significance |
| --- | --- | --- | --- | --- | --- |
| Beta | pb(age) | -0.00795 | LRT (drop1) | 0.00004 | \*\*\* |
| Beta | sex (sex2) | -0.0989 | LRT (drop1) | 0.222 | ns |
| Weibull | pb(age) | -0.00191 | Wald t | 0.00019 | \*\*\* |
| Weibull | sex (sex2) | -0.0145 | Wald t | 0.459 | ns |

Age has a significantly negative effect on accuracy, sex has no significant effect; this is consistent with the findings of Vallat & Walker (2021), Bechny et al. (2025) and Rossi et al. (2026).

The Weibull model has a lower AIC (-179.53 vs -174.18 for Beta), indicating a somewhat better statistical fit. The Weibull, however, is not bounded to the interval (0,1) by construction as the Beta is; a check shows that the mean predicted probability of accuracy exceeding 1 is 0.0000. The Beta remains the more principled choice for reporting. The effect of age is practically linear (edf = 2.00).

Beta model prediction (women, sex=1): for a 30-year-old the expected accuracy is 74.7%, for a 90-year-old 64.7%, a drop of 10.0 percentage points over 60 years of age.

&#91;embedded content: per\_subject\_accuracy.csv and age\_curve.csv (subject-number column omitted) · 78 subjects, Beta GAMLSS\]

The curves for women (F) and men (M) almost coincide, consistent with the finding that sex has no significant effect. The dots show that the spread among subjects is large in every age group, so age explains only part of the differences.

### 4.5 Clinical markers

Mean difference (YASA - human) in derived PSG markers:

| Marker | Mean diff (YASA - human) | SD | 95% CI (subject-level bootstrap) |
| --- | --- | --- | --- |
| TST (min) | -7.8 | 40.5 | \[-15.5, -0.2\] |
| Sleep efficiency (%) | -1.5 | 7.7 | \[-3.0, -0.1\] |
| REM latency (min) | -68.8 | 60.5 | \[-80.1, -57.6\] |
| WASO (min) | +30.9 | 38.0 | \[+24.0, +37.7\] |

Median values for the human scorer: TST 414 min, SE 83%, WASO 42 min, REM latency 74 min.

**REM latency, YASA deviation, cause not established:** a difference of -69 min (rem\_latency\_check.py). Of the epochs that the human scored as wake in bed before falling asleep (38.6 h in total), YASA labels 14.4% as REM and 10.5% as N2. The first sleep epoch from YASA is REM in 56 of 142 recordings, and YASA's REM latency is below 5 min in 88 recordings (median 1.5 min, versus 74 min for the human). The wake-to-REM confusion amounts to about 2 min per recording (14.4% of 38.6 h over 142 recordings), so on its own it probably does not explain the full 69-minute difference. REM latency from YASA should not be used as a clinical marker without human verification.

### 4.6 Worst recordings

The three recordings with the lowest accuracy (37-44%) were examined through per-stage confusion matrices (rem\_latency\_check.py). The error patterns differ: in the first, YASA systematically labels N2 as N3 (261 epochs; YASA gives 40% N3 versus 11% for the human); in the second, N2 as REM or wake (243 and 171 epochs); and in the third, wake as REM (144 epochs). For two of the three recordings the same subject's other night also exists, and its accuracy is ordinary (0.66 and 0.72). This points to a night-specific problem (e.g. signal quality or electrode placement) rather than a property of the subject. Visual inspection of the raw signal for these nights has not been done.

## 5. Limitations and Known Gaps

- **Race/ethnicity:** not available in the Sleep-EDF dataset. This remains an open gap that the authors of Rossi et al. (2026) also explicitly acknowledged; addressing it would require an external dataset with documented racial/ethnic labels, which is beyond the scope of this exercise.
- **AHI/PLMI:** not available in this dataset, so clinical bias could not be examined.
- **Channels:** a single EEG channel (Fpz-Cz) was used instead of a central derivation (e.g. C4-M1) for which YASA was developed. The effect on the results has not been assessed.
- **Scoring rules:** the reference hypnograms were scored according to Rechtschaffen and Kales (1968), not AASM rules. Differences in the definition of some stages (e.g. N1, N3) may affect accuracy; this effect has not been assessed.
- **Analysis window:** the 2 h threshold for the end of the main sleep period was chosen based on the duration of night awakenings in this dataset (the longest about 1.5 h), and the window is determined from the human hypnogram. The window was therefore tuned after seeing these data and would not be available in real use.
- **Dataset size:** 142 recordings (78 subjects) is much smaller than the BSWR dataset (4,075 recordings in Bechny et al., 2025), so estimates are less precise.
- **Considered but not performed:** re-running YASA without the EMG channel (EEG and EOG only) to compare REM latency and accuracy. Assumption (not verified): YASA computes time- and frequency-domain features from a signal downsampled to 100 Hz, whereas here a 1 Hz EMG envelope is used, so it is possible that this contributes to the short REM latency.
- **Version mismatch:** YASA's model was trained with scikit-learn 0.24.2, while the environment uses 1.9.1 (InconsistentVersionWarning). Predictions were not compared against a compatible version.

## 6. Coverage of Requirements by This Exercise

| Requirement | What was done | Status |
| --- | --- | --- |
| Article 15: accuracy and robustness | Accuracy, kappa and F1 with subject-level bootstrap CIs on a dataset outside training; robustness (noise, poor signal) not tested | Partially addressed |
| Article 10: bias by age and sex | GAMLSS on 78 subjects, one cohort | Addressed (age and sex only) |
| Article 10: other relevant subgroups (race/ethnicity) | Data not available in this dataset | Not addressed |
| Annex IV 2(g): discriminatory impact, clinical factors (AHI/PLMI) | Data not available in this dataset | Not addressed |
| Documented data flow | 153 -> 142 recordings, reasons in excluded\_recordings.csv | Addressed |
| Reproducibility | Versions, environment and seed recorded; commit recorded; results not in the repository | Partially addressed |
| Post-market monitoring plan | Written as a plan, not in production | Addressed (as a plan only) |
| Data minimization | Aggregates only, no IDs | Addressed |

The table shows which requirements this exercise touched and where the gaps are: race/ethnicity, clinical factors (AHI/PLMI), robustness tests and archiving of results outside the repository. It makes no statement on whether YASA complies with the EU AI Act.

## 7. Post-Market Monitoring Plan

If YASA (or a similar model) were part of a real medical product, the following should be monitored continuously after launch:

- **Age distribution of users in production:** if the share of patients over 80 rises relative to the validation set, expect a further drop in accuracy (based on the GAMLSS curve)
- **Racial/ethnic distribution:** collect these data prospectively as soon as possible, to close the current gap
- **Rate of manual corrections by sleep stage:** monitor N1 in particular (F1=0.281), where the model is weakest; a sudden rise in corrections may signal drift
- **Early REM at sleep onset:** in this audit YASA labeled the first sleep epoch as REM in 56 of 142 recordings; monitor the share of such early REM epochs and REM latency by cohort, and do not use REM latency as a clinical marker without human verification
- **Signal quality per night:** the worst recordings in this audit (accuracy 37-44%) had an ordinary second night from the same subject, which points to a night-specific problem; introduce automatic signal-quality checks and flag suspicious nights for manual review
- **Calibration of confidence scores:** whether predict\_proba outputs still correctly reflect actual reliability on new data
- **Clinical markers (TST/WASO/REM latency) by cohort:** systematic deviations by patient group (e.g. clinic or device) would signal dataset shift
- **Periodic re-validation** (e.g. annually or on model version change) on a fresh, independent sample

## 8. Reproducibility

Software versions and hardware were recorded by the script reproducibility\_info.py (output: reproducibility\_info.txt), and the whole conda environment was exported to environment.yml. GAMLSS is called from Python via rpy2.

| Component | Version |
| --- | --- |
| Python | 3.11.16 |
| YASA | 0.7.0 |
| MNE | 1.13.2 |
| scikit-learn | 1.9.1 |
| LightGBM | 4.7.0 |
| NumPy / pandas | 2.4.6 / 3.0.5 |
| rpy2 | 3.5.11 |
| R | 4.1.3 |
| gamlss / gamlss.dist | 5.4.22 / 6.1.1 |
| OS / hardware | Windows 11 (build 26300), Intel x64, 8 logical cores |

**Random seed:** bootstrap\_ci.py uses seed 20261002. The YASA prediction and the gamlss RS algorithm are deterministic: re-running the whole pipeline gave byte-identical results.

**Git:** commit 8bd916223e083001b4c5e7193b84f1d84e43660b (Initial commit: YASA sleep staging audit on Sleep-EDF Expanded), repository https://github.com/marijaskaric/sleep\_staging\_ai\_audit. The repository contains scripts that compare YASA automatic sleep staging with human scoring; data and results live outside the repository, in the folder ../Sleep Cassette/. Raw EDF files are downloaded from PhysioNet (Kemp et al., 2000).

**Remaining for a full reproducibility package:** results and data are not in the repository, so they should be archived separately or the way to obtain them should be documented.

## 9. Conclusion

On the available data (age and sex), YASA shows a statistically significant, moderate decline in accuracy with age (74.7% at age 30 versus 64.7% at age 90, Beta GAMLSS) and no significant effect of sex. This matches what Vallat & Walker (2021) and Bechny et al. (2025) reported, here on an independent dataset.

Two requirement areas were not addressed because the data were missing: examination of other relevant subgroups such as race/ethnicity (Article 10) and part of the discriminatory-impact metrics (Annex IV, point 2(g)), which also needs clinical indices (AHI/PLMI). A real manufacturer would have to cover these before CE marking. The reason for the large REM-latency difference remains unknown.

**This was a practice exercise, not a regulatory submission.** Whether the AI Act and MDR obligations would apply depends on whether a final product would need a notified body under the MDR (section 2.2); for R&D before placing on the market, Article 2(8) of the AI Act excludes most of them.

## 10. References

Bechny, M., Fiorillo, L., van der Meer, J., Schmidt, M., Bassetti, C., Tzovara, A., & Faraci, F. (2025). Beyond accuracy: A framework for evaluating algorithmic bias and performance, applied to automated sleep scoring. *Scientific Reports, 15*, Article 21421. https://doi.org/10.1038/s41598-025-06019-4

Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement, 20*(1), 37-46.

Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *The Annals of Statistics, 7*(1), 1-26.

European Parliament & Council of the European Union. (2017). *Regulation (EU) 2017/745 on medical devices (MDR)*. https://eur-lex.europa.eu/eli/reg/2017/745/oj

European Parliament & Council of the European Union. (2024). *Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act)*. https://eur-lex.europa.eu/eli/reg/2024/1689/oj

European Parliament & Council of the European Union. (2026). *Regulation (EU) 2026/1744* \[Digital Omnibus on AI; published in the Official Journal of the EU on 24 July 2026\]. https://eur-lex.europa.eu/eli/reg/2026/1744/oj/eng

Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., Mietus, J. E., Moody, G. B., Peng, C.-K., & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation, 101*(23), e215-e220.

Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D., Brodbeck, C., Goj, R., Jas, M., Brooks, T., Parkkonen, L., & Hämäläinen, M. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience, 7*, Article 267.

Johner Institute. (n.d.). *AI guideline for medical device manufacturers* \[Source code repository\]. GitHub. https://github.com/johner-institut/ai-guideline

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems, 30*.

Kemp, B., Zwinderman, A. H., Tuk, B., Kamphuisen, H. A. C., & Oberye, J. J. L. (2000). Analysis of a sleep-dependent neuronal feedback loop: The slow-wave microcontinuity of the EEG. *IEEE Transactions on Biomedical Engineering, 47*(9), 1185-1194. (Sleep-EDF Expanded dataset: https://physionet.org/content/sleep-edfx/1.0.0/)

Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized additive models for location, scale and shape. *Journal of the Royal Statistical Society: Series C (Applied Statistics), 54*(3), 507-554.

Rossi, A. D., Metaldi, M., Bechny, M., Filchenko, I., van der Meer, J., Schmidt, M. H., Bassetti, C. L. A., Tzovara, A., Faraci, F. D., & Fiorillo, L. (2026). SLEEPYLAND: Trust begins with fair evaluation of automatic sleep staging models. *npj Digital Medicine, 9*, Article 55. https://doi.org/10.1038/s41746-025-02237-2

Vallat, R., & Walker, M. P. (2021). An open-source, high-performance tool for automated sleep staging. *eLife, 10*, Article e70092. https://doi.org/10.7554/eLife.70092

## Appendix A: Deviations and Methodology History

The pipeline went through three iterations before the final version. The table shows how results changed as methodological errors were corrected.

| Metric | Whole recording (v0) | Night +-30 min (v1) | Final (v2) |
| --- | --- | --- | --- |
| Accuracy (mean) | 0.808 | 0.711 | 0.692 |
| Kappa (mean) | 0.641 | 0.589 | 0.569 |
| TST difference (YASA - human, min) | +85 | +8 | -8 |
| WASO difference (min) | +482 | +21 | +31 (SD 38) |
| REM latency difference (min) | -90 | -129 | -69 |
| Windows >12 h | - | 36 | 3 |

From v0 to v2: v0 analyzed the whole recording, so daytime wake inflated accuracy and WASO; v1 used a window of +-30 min around sleep; v2 uses a window from lights-off to the end of the main sleep period plus 30 min (section 3.4). This cut the number of implausible windows (>12 h) from 36 to 3. In v2 the GAMLSS was also moved from 142 nights to 78 subjects (nights averaged) to avoid pseudo-replication, and the tests changed from t-tests to the likelihood-ratio test. All of these changes were made after seeing the results (see section 3.6).

## Appendix B: Scripts and Output Files

| File | Role |
| --- | --- |
| yasa\_sleepedfx\_pilot.py | Main script: YASA predictions, window definition, metrics and clinical markers |
| gamlss\_real\_analysis.py | GAMLSS analysis of the effect of age and sex (Beta and Weibull) |
| bootstrap\_ci.py | Subject-level bootstrap confidence intervals |
| rem\_latency\_check.py | REM latency diagnostics and confusion matrices of the worst recordings |
| reproducibility\_info.py | Record of software and hardware versions |
| generate\_report\_via\_api.py | Generation of the report text (sends aggregates only) |
| excluded\_recordings.csv | Omitted recordings with reasons |
| yasa\_sleepedfx\_clinical\_stats.csv | Clinical parameters per recording |
| reproducibility\_info.txt, environment.yml | Software versions and exported conda environment |
