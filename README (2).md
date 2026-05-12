# Gun Violence Early Warning System

Predicting county-level firearm death rate spikes in the United States using socioeconomic, demographic, and social vulnerability features.

**Team:** Josh, Jake, Sebastian, Jessica

---

## Overview

This project builds a binary classification model to predict whether a U.S. county will experience a significant year-over-year spike in its firearm crude death rate. A spike is defined as a **>25% increase** in the crude rate relative to the prior year. The goal is to give public agencies and policymakers an early signal to allocate resources proactively before violence escalates.

---

## Problem Statement

Gun violence is one of the leading causes of preventable death in the United States. Public agencies often respond reactively because they lack tools to identify which counties are at elevated risk before violence occurs. This model uses publicly available socioeconomic and health data to flag high-risk counties one year in advance.

**Primary question:** Given socioeconomic, demographic, and social vulnerability features for a U.S. county, can we predict whether that county will experience a spike in gun violence the following year?

**Secondary question:** Which features are most predictive, and what do they tell us about intervention priorities?

---

## Data Sources

| Source | Description | Years |
|--------|-------------|-------|
| [CDC WONDER](https://wonder.cdc.gov/) — Underlying Cause of Death (Single Race) | County-level firearm death counts, population, and crude rate per 100k | 2018–2024 |
| [CDC Social Vulnerability Index (SVI)](https://www.atsdr.cdc.gov/placeandhealth/svi/index.html) | 20+ socioeconomic and demographic vulnerability indicators by county | 2018, 2020 |
| [U.S. Census ACS 5-Year Estimates](https://www.census.gov/data/developers/data-sets/acs-5year.html) | Poverty rate, unemployment rate, and median household income by county | 2018–2023 |

---

## Repository Structure

```
├── gun_violence_final_project.ipynb         # Main pipeline: data ingestion, cleaning, EDA
├── MLfinal_LogisticRegression.ipynb         # Logistic regression: tuning, leakage fix, SHAP
├── gun_violence_master.csv                  # Merged dataset (pre-cleaning)
├── gun_violence_final.csv                   # Final clean dataset used for modeling
├── eda_overview.png                         # EDA visualization output
├── SVI_2018_US_county.csv                   # SVI data (2018)
├── SVI_2020_US_county.csv                   # SVI data (2020)
└── Underlying Cause of Death, 2018-2024, Single Race (2).csv   # CDC WONDER export
```

---

## Setup & Requirements

**Python 3.8+**

Install dependencies:

```bash
pip install pandas numpy requests openpyxl matplotlib seaborn scikit-learn
```

**Census API Key**

Cell 6 pulls ACS data from the Census Bureau API. You'll need a free API key:

1. Register at [api.census.gov/data/key_signup.html](https://api.census.gov/data/key_signup.html)
2. Replace the placeholder in Cell 6:
   ```python
   CENSUS_API_KEY = 'your_key_here'
   ```

---

## Notebook Walkthrough

### Cell 1 — Install & Import
Installs required libraries and imports core packages.

### Cell 2 — Load CDC WONDER Data
Loads the CDC WONDER death records CSV. Strips footer metadata rows, renames columns, imputes CDC-suppressed death counts (counts < 10) as 5, recalculates crude rates per 100k, and flags suppressed rows.

### Cell 3 — Build Target Variable
Computes a lagged crude rate per county and calculates the year-over-year percentage change. Labels each county-year observation as a **spike (1)** if the crude rate increased by more than 25%, or **no spike (0)** otherwise. Rows without a prior-year observation (i.e., the first year per county) are dropped.

### Cell 4 — Load SVI Data
Loads CDC SVI snapshots for 2018 and 2020. Extracts 20 vulnerability features including unemployment, uninsured rate, disability, minority share, single-parent households, housing stress, and composite theme scores. Replaces CDC's missing value flag (−999) with NaN.

### Cell 5 — Merge SVI + WONDER
Assigns each WONDER year to the closest available SVI snapshot (2019 → 2018 SVI; 2020 onward → 2020 SVI) and merges on county FIPS code and SVI year.

### Cell 6 — Pull ACS Data via API
Pulls five ACS variables for each year 2018–2023 using the Census Bureau API: total population, population below poverty, civilian labor force, unemployed, and median household income. Derives poverty rate, unemployment rate, and median income as final features.

### Cell 7 — Merge ACS into Main Dataset
Joins ACS data onto the main dataset by FIPS and year, matching on exact year (ACS is annual).

### Cell 8 — Final Cleanup & Save
Drops 2024 rows (no matching ACS data), removes leakage columns, renames all SVI feature columns for readability, and saves `gun_violence_master.csv`.

### Cell 9 — Handle Missingness & Finalize
Drops `svi_per_capita_income` (too many missing due to a 2018 SVI schema change), imputes `svi_housing_burden` with the column median, drops the small number of remaining rows with any missing values (~8–20 rows), and saves the final clean dataset as `gun_violence_final.csv`.

### Cell 10 — EDA: Descriptive Statistics
Loads the final dataset and prints an overview: row count, county count, years covered, target distribution, and descriptive statistics for all feature columns.

### Cell 11 — EDA: Visualizations
Produces a 2×3 panel saved as `eda_overview.png` covering:
- Target variable class distribution
- Crude rate histogram (clipped at 200)
- Crude rate by spike label (box plot)
- Spike rate by year
- SVI composite score vs. crude rate (scatter)
- Pearson correlation of each feature with the spike target

---

## Feature List

After cleaning, the model uses **26 features** across three categories:

**SVI Features (CDC Social Vulnerability Index)**
| Feature | Description |
|---------|-------------|
| `svi_unemp_rate` | Unemployment rate % |
| `svi_no_hs_diploma` | Population without high school diploma % |
| `svi_uninsured` | Uninsured population % |
| `svi_age65_plus` | Population aged 65+ % |
| `svi_age17_under` | Population under 17 % |
| `svi_disabled` | Disabled population % |
| `svi_single_parent` | Single-parent households % |
| `svi_minority` | Minority population % |
| `svi_limited_english` | Limited English proficiency % |
| `svi_multi_unit` | Multi-unit housing % |
| `svi_mobile_homes` | Mobile homes % |
| `svi_crowded_housing` | Crowded housing % |
| `svi_no_vehicle` | Households with no vehicle % |
| `svi_group_quarters` | Group quarters population % |
| `svi_poverty_rate` | Population below 150% poverty line % |
| `svi_housing_burden` | Housing cost burden % |
| `RPL_THEME1` | Socioeconomic vulnerability composite score |
| `RPL_THEME2` | Household composition composite score |
| `RPL_THEME3` | Minority/language composite score |
| `RPL_THEME4` | Housing/transportation composite score |
| `RPL_THEMES` | Overall SVI composite score |

**ACS Features (U.S. Census)**
| Feature | Description |
|---------|-------------|
| `acs_poverty_rate` | Population below poverty line % |
| `acs_unemp_rate` | Civilian unemployment rate % |
| `acs_median_income` | Median household income ($) |

**Target Variable**
| Column | Description |
|--------|-------------|
| `spike` | 1 = crude rate increased >25% YoY, 0 = otherwise |

---

## Target Variable Notes

- The spike threshold of **25% year-over-year increase** was chosen to capture meaningful escalation events rather than normal statistical fluctuation.
- CDC suppresses death counts below 10 to protect privacy; suppressed values are imputed as 5 (the midpoint of the 1–9 range), which slightly underestimates crude rates for small counties.
- The final dataset covers **2019–2023** (2018 is dropped as it has no prior-year lag; 2024 is dropped due to missing ACS data).

---

## Candidate Models

Four classifiers are evaluated:

| Model | Rationale |
|-------|-----------|
| Logistic Regression | Interpretable baseline; handles multicollinearity via regularization |
| Random Forest | Handles high-dimensional data and nonlinear feature interactions |
| XGBoost | Strong tabular data performer; gradient boosting with built-in regularization |
| Neural Network | Exploratory; tests for deep feature interactions tree models may miss |

**Performance metrics:** Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix

---

## Limitations

- **Binary label:** The median-split approach discards within-group variation; a regression target (predicting the actual crude rate or change) could be more informative.
- **County granularity:** County-level predictions mask neighborhood-level variation within a county.
- **Unmeasured factors:** Specific legislation, local policing policy, and executive orders cannot be captured as features.
- **Evaluation split:** A stratified random split was used; a temporal split (train 2019–2021, test 2022–2023) would better simulate real deployment conditions.
- **SVI staleness:** SVI snapshots are only available for 2018 and 2020; years 2021–2023 use the 2020 SVI, which may not reflect post-pandemic shifts in vulnerability.

---

## Logistic Regression Notebook (`MLfinal_LogisticRegression.ipynb`)

This notebook contains the full logistic regression implementation, including hyperparameter tuning, leakage detection, and model explainability. It picks up from `gun_violence_final.csv` produced by the main pipeline.

### Modeling approach

The model is built as a `sklearn` Pipeline with two steps: `StandardScaler` → `LogisticRegression`. Standardization is required because elastic net regularization is sensitive to feature scale — without it, coefficients for features measured in different units (e.g., percentages vs. dollar income) are penalized unevenly.

**Regularization:** Elastic net (`penalty="elasticnet"`, `solver="saga"`) is used instead of pure L1 or L2 to get the best of both: L1 drives sparse coefficients (zeroing out irrelevant features) while L2 stabilizes correlated ones. The `l1_ratio` and `C` parameters are tuned via grid search.

**Class imbalance:** `class_weight="balanced"` is set so the model does not learn to predict the majority class by default.

**Train/test split:** A `GroupShuffleSplit` on FIPS ensures that all observations for a given county land entirely in either train or test — preventing the model from seeing future data for a county it was trained on.

### Cell walkthrough

**Cell 1 — Threshold sensitivity analysis**
Tests the top-5 most important features across multiple alternative labeling thresholds (different crude rate cutoffs for "high" violence). Finds features that appear in the top 5 regardless of threshold choice — these are the most robust predictors and the most defensible to report.

**Cell 2 — Leakage fix**
Identifies `population` as a leaky feature: since `crude_rate = (deaths / population) × 100,000`, including population gives the model indirect access to the target. Drops it and reruns the full pipeline using the best hyperparameters from the prior grid search.

**Cell 3 — Comparison table + SHAP**
Produces a side-by-side comparison of the original (leaky) model vs. the fixed model across Train Accuracy, Test Accuracy, train/test gap, F1, and ROC-AUC. Then runs SHAP (`LinearExplainer`) on the fixed model to produce:
- **Beeswarm plot** — shows the direction and magnitude of each feature's effect on individual predictions
- **Bar plot** — ranks features by mean absolute SHAP value (global importance)

### Key design decisions

| Decision | Rationale |
|----------|-----------|
| Elastic net over L1/L2 | Handles correlated SVI features better than pure lasso; L2 component prevents coefficient instability |
| `saga` solver | Only solver that supports elastic net in scikit-learn |
| `GroupShuffleSplit` on FIPS | Prevents county-level data leakage across the train/test boundary |
| SHAP `LinearExplainer` | Exact (not approximate) SHAP values for linear models; fast and interpretable |
| Threshold sensitivity check | Validates that key features are stable across labeling choices, not artifacts of a single cutoff |

### Additional dependency

```bash
pip install shap
```

---

## Live Demo

[https://gun-violence-early-warning-system.streamlit.app/](https://gun-violence-early-warning-system.streamlit.app/)
