# STAI-X Challenge 2026

This project predicts suspected nonfatal overdose emergency department (ED) visit rates across jurisdictions and periods: `rate_per_10000_ed_visits`.

The workflow includes data merging, alternative weather-missingness strategies, five-fold period-grouped model comparisons, and interpretable MAT density image features. The latest submission candidate is **a pooled model with Dataset C and image features, combining tuned HistGradientBoosting and RandomForest predictions with weights of 0.7 and 0.3**.

The first submission used a single HistGradientBoosting model. The historical development notes describe the later ensemble as awaiting leaderboard validation. Its local validation score should not be confused with the first submission's Public Leaderboard score.

## Project Structure

```text
stai-x-challenge-2026/
|-- train/                         # Official training inputs, including MAT images
|-- val/                           # Official prediction inputs; outcomes hidden
|-- outputs/                       # Intermediate datasets and comparison results
|-- notebooks/                     # Preparation, experiments, and submission code
|   |-- statx_helpers.py           # Shared development utilities
|   |-- kaggle_submission.py       # Self-contained submission script
|   `-- kaggle_submission.ipynb    # Self-contained submission notebook
|-- Data_Description.md            # Official dataset documentation
`-- sample_submission.csv          # Official prediction template
```

The raw-input paths describe the expected local or Kaggle layout. Obtain the official competition inputs through Kaggle; do not redistribute competition data.

## Completed Work

1. Joined labels and covariates on `period_id` and `jurisdiction`.
2. Built pooled (universal) datasets and separate datasets for each scored category.
3. Compared three weather-processing strategies: A, B, and C.
4. Replaced the initial single 80/20 period split with five-fold `GroupKFold` by period for model comparison.
5. Extracted eight interpretable MAT image features and evaluated their incremental predictive value.
6. Compared linear models, tree models, category-specific models, text representations, multitask training, and ensembles.
7. Generated `submission.csv` and recorded the first single-model submission.
8. Packaged the later tuned ensemble as a self-contained notebook for competition re-execution.

## Historical Submission Record

| Date | Configuration | Public LB MAE | Local five-fold MAE | Rank at the time |
|---|---|---:|---:|---|
| 2026-06-13 | Universal + Dataset C + images + HistGB | 1.328 | 1.725 | 67 / 75 |

These are the values recorded in the project notes. The rank is a snapshot, not a verified final rank. The Public Leaderboard result belongs to the first single-model entry, not the later ensemble.

The Public Leaderboard MAE was lower than local CV. Different splits or scoring conditions can contribute to this difference; the numbers alone do not establish why the holdout was easier or prove the precise scoring formula.

This is a **code competition**: the submitted notebook must rebuild the pipeline and write a valid prediction file when re-executed on a later hidden window. Submitting a fixed CSV alone does not provide that reproducible implementation.

The later candidate uses **tuned HistGB with absolute-error loss plus RandomForest, weighted 0.7/0.3**. The recorded local five-fold MAE is **1.7086**, compared with **1.7125** for equal-weight averaging and **1.7247** for default single-model HistGB. Its final Kaggle selection and leaderboard result are not established by these notes.

## Development History

### Initial Team Data and Modeling Framework

- `01_prepare_merged_data.ipynb`: creates pooled and category-specific merged tables.
- `02` and `03`: create A/B/C datasets for pooled and category-specific modeling.
- `statx_helpers.py`: shared period splitting, dataset construction, preprocessing, estimators, and evaluation.
- `04` and `05`: comparison framework for the mean baseline, Ridge, ElasticNet, HistGB, and RandomForest.

### Image Features Branch

- `06_image_features.py`: converts each MAT heatmap into eight interpretable image descriptors and writes `outputs/image_features.csv`.
- `07_validate_image_features.py`: checks feature integration and compares HistGB with and without images.
- `08_image_feature_gains.py`: quantifies image gains for each model and weather strategy.
- `09_text_features.py`: refactors the teammate's earlier `STAIX-26.py`. It replaces hard-coded paths, corrects validation-data usage, aligns train/validation keyword definitions, removes a duplicated overdose term and unused code, and saves reusable features.
- `10_text_feature_gains.py`: compares covariates, text, images, and their combination.
- `11_make_submission.py`: generates the original single-HistGB prediction file.
- `12_ensemble_validation.py`: evaluates equal-weight HistGB/RandomForest averaging.
- `13_multitask_categories.py`: evaluates adding the five non-scored categories as training rows.
- `14_tuning_and_loss.py`: evaluates MAE-aligned loss, modest hyperparameter tuning, and ensemble weights.
- `15_text_tfidf.py`: evaluates fold-fitted TF-IDF with TruncatedSVD.
- `kaggle_submission.py` and `kaggle_submission.ipynb`: inline the candidate's preprocessing, image extraction, modeling, and output logic.

The comparison scripts were updated to use `cross_val_by_period`, with weather reference statistics recalculated inside each fold. Archived comparison tables remain under `outputs/`.

## Validation Design

A period contains multiple states and drug categories. Random row splitting can place observations from the same period on both sides of a split, producing an overly optimistic estimate for unseen-period prediction.

The development workflow uses **five-fold GroupKFold grouped by `period_id`**, implemented by `statx_helpers.cross_val_by_period`:

- Each period belongs to exactly one held-out fold.
- No period is shared between training and validation within a fold.
- All 77 training periods are evaluated once.
- The comparison helper reports mean RMSE, MAE, and R-squared across folds.
- Weather processing for B and C is recalculated with that fold's training reference.

The original fixed 80/20 period split and its saved datasets remain available. That split contains 61 training periods and 16 held-out periods. It is a development split, distinct from the official `val/` prediction inputs.

Grouping by opaque period IDs does **not** constitute chronological forward validation. The later tuning script pools out-of-fold predictions before calculating MAE, whereas the shared comparison helper averages fold metrics. Small differences between these summaries should be interpreted accordingly.

## Notebook and Script Guide

Run development notebooks in numerical order where their saved inputs are required.

| File | Purpose | Main output or result |
|---|---|---|
| `01_prepare_merged_data.ipynb` | Read official inputs and merge labels/covariates | Pooled and category-specific merged tables |
| `02_make_universal_period_datasets.ipynb` | Build pooled A/B/C development datasets | `universal_*_train.csv`, `universal_*_val.csv` |
| `03_make_category_period_datasets.ipynb` | Build separate A/B/C datasets for each category | Category-specific train/validation tables |
| `04_compare_universal_period_models.ipynb` | Compare weather strategies and five models using grouped CV | `universal_period_model_comparison.csv` |
| `05_compare_category_period_models.ipynb` | Compare models separately for each scored category | `category_period_model_comparison.csv` |
| `06_image_features.py` | Extract image descriptors once per unique state-period and split | `image_features.csv` |
| `07_validate_image_features.py` | Validate joins and image gains for HistGB | Grouped-CV comparison |
| `08_image_feature_gains.py` | Evaluate image gains across all models and A/B/C | `image_feature_gains.csv` |
| `09_text_features.py` | Extract keyword counts, length, release indicators, and risk labels | `text_features.csv` |
| `10_text_feature_gains.py` | Evaluate covariates/text/images combinations under Dataset C | `text_feature_gains.csv` |
| `11_make_submission.py` | Fit the original single HistGB on all labeled training data | Root `submission.csv` |
| `12_ensemble_validation.py` | Evaluate equal-weight HistGB + RF | Recorded MAE 1.7125 |
| `13_multitask_categories.py` | Compare three-category and eight-category training | Extra categories not retained |
| `14_tuning_and_loss.py` | Compare loss, tuning, and blend weights | Tuned 0.7/0.3 candidate |
| `15_text_tfidf.py` | Evaluate fold-fitted TF-IDF/SVD | Not retained |
| `STAIX-26.py` | Earlier teammate text prototype | Historical reference; superseded by 09 |
| `kaggle_submission.ipynb` | Self-contained competition candidate | `/kaggle/working/submission.csv` |
| `kaggle_submission.py` | Script equivalent of the submission notebook | Same pipeline |

### Dataset Sizes

The pooled modeling table contains the three scored categories and includes `overdose_category` as an input:

| Development dataset | Rows | Periods |
|---|---:|---:|
| Full pooled table | 11,781 | 77 |
| Fixed-split pooled training table | 9,333 | 61 |
| Fixed-split pooled validation table | 2,448 | 16 |
| Full table for each scored category | 3,927 | 77 |
| Fixed-split training table for each category | 3,111 | 61 |
| Fixed-split validation table for each category | 816 | 16 |

The three scored categories are `all_drugs`, `all_opioids`, and `all_stimulants`. They overlap and must not be summed.

## Shared Utilities

`notebooks/statx_helpers.py` centralizes:

- `handle_text()`: fills missing releases and constructs `has_doh_release`.
- `split_by_period()`: creates a development split using whole periods.
- `make_dataset()`: builds an A/B/C dataset using a specified reference.
- `save_all_datasets()`: saves the fixed-split modeling tables.
- `build_preprocessor()`: numeric scaling and categorical encoding.
- `get_models()`: defines the comparison estimators.
- `fit_and_score_dataset()`: fits models and reports RMSE, MAE, and R-squared.
- `cross_val_by_period()`: runs grouped five-fold comparisons with fold-specific weather processing.

The self-contained submission pipeline additionally uses numeric median imputation and handles unknown categories.

## Weather Strategies

### Dataset A: No Weather

Filename suffix: `A_no_weather`.

Drop `temp_avg_f` and `precip_in`. This avoids assumptions about their missing values but discards weather information.

### Dataset B: Period Median

Filename suffix: `B_weather_period_median`.

Keep weather variables. Fill missing values from training-reference period medians and then global training medians. Include a `weather_missing` indicator.

### Dataset C: Similar States

Filename suffix: `C_weather_similar_state`.

Keep weather variables. For AK, HI, and DC, first try designated similar states in the same period, then training-reference period medians, then global training medians. Include `weather_missing`.

Donor groups:

```python
SIMILAR_STATES = {
    "AK": ["WA", "MT", "ND", "MN"],
    "HI": ["CA", "FL"],
    "DC": ["MD", "VA"],
}
```

Validation processing uses the training reference. For a wholly unseen period, no same-period reference donors exist, so the global-median fallback can apply. The candidate's missingness flag is derived from temperature missingness.

## MAT Image Features

Each `{JURISDICTION}_{PERIOD_ID}.png` is processed once per unique split/state/period, not once per drug category. Near-black pixels with `max(R, G, B) < 20` are masked before density summaries are calculated.

The eight features are:

1. Background ratio.
2. Mean luminance within the mask.
3. Luminance standard deviation.
4. Maximum luminance.
5. The 90th luminance percentile.
6. Fraction above the high-density luminance threshold.
7. Top-decile intensity share.
8. Normalized density-weighted spatial spread.

The output includes a `split` column identifying the competition training or validation directory. Features join back on `period_id` and `jurisdiction`.

The project chose compact interpretable descriptors rather than training a CNN. Early exploratory notes recorded weak individual feature correlations (absolute Spearman correlation no greater than 0.22) and state-related structure. This motivated a lower-complexity representation; it does not establish that a CNN could never help.

Luminance is a proxy for the rendered viridis map, not a calibrated prescriber count. Background ratio still captures map coverage and state geometry.

## Model Comparisons

The comparison framework includes:

- Model 0: category-mean baseline.
- Model 1: Ridge fixed effects.
- Model 2: ElasticNet fixed effects.
- Model 3: HistGradientBoosting.
- Model 4: RandomForest.

Numeric inputs use `StandardScaler`; categorical inputs use `OneHotEncoder`. Raw `state_doh_release` text is excluded from the selected model, while `has_doh_release` is retained.

### Historical Pooled Five-Fold Results

Selected rows from `outputs/universal_period_model_comparison.csv`:

| Dataset | Model | RMSE | MAE | R-squared |
|---|---|---:|---:|---:|
| B | HistGB | 2.670 | 1.732 | 0.926 |
| C | HistGB | 2.672 | 1.732 | 0.925 |
| C | RandomForest | 2.786 | 1.815 | 0.919 |
| A | HistGB | 2.919 | 1.944 | 0.911 |
| B/C | ElasticNet | 4.001 | 2.704 | 0.833 |
| All strategies | Mean baseline | 6.426 | 4.115 | 0.569 |

Tree models performed better than the tested linear models. B and C were nearly tied and both outperformed A. Dataset C was retained as the development choice; this does not demonstrate that its donor-state assumptions are correct for every jurisdiction.

### Image Feature Gains

Historical five-fold comparisons from `outputs/image_feature_gains.csv`. Positive improvement means lower error.

| Dataset | Model | MAE without images | MAE with images | MAE improvement | RMSE improvement |
|---|---|---:|---:|---:|---:|
| A | Mean baseline | 4.115 | 4.115 | +0.0% | +0.0% |
| A | Ridge | 3.099 | 3.091 | +0.2% | +0.2% |
| A | ElasticNet | 2.901 | 2.898 | +0.1% | +0.1% |
| A | HistGB | 1.944 | 1.913 | +1.6% | +1.5% |
| A | RandomForest | 1.924 | 1.834 | +4.7% | +4.3% |
| B | Mean baseline | 4.115 | 4.115 | +0.0% | +0.0% |
| B | Ridge | 3.102 | 3.096 | +0.2% | +0.1% |
| B | ElasticNet | 2.704 | 2.706 | -0.1% | +0.0% |
| B | HistGB | 1.732 | 1.723 | +0.5% | +0.4% |
| B | RandomForest | 1.817 | 1.760 | +3.1% | +2.9% |
| C | Mean baseline | 4.115 | 4.115 | +0.0% | +0.0% |
| C | Ridge | 3.096 | 3.090 | +0.2% | +0.1% |
| C | ElasticNet | 2.705 | 2.704 | +0.0% | +0.1% |
| C | HistGB | 1.732 | 1.725 | +0.4% | +0.2% |
| C | RandomForest | 1.815 | 1.758 | +3.2% | +2.9% |

RF showed the largest relative image gain, but HistGB still had the lower absolute MAE. Under Dataset C, HistGB reached 1.7247 and RF reached approximately 1.758. The tested tree models improved across all three weather strategies, whereas linear-model changes were small.

### Text Features

The keyword pipeline extracts crisis, warning, and response counts, release length, release-presence indicators, and risk labels.

Historical Dataset C MAE from `outputs/text_feature_gains.csv`:

| Model | Covariates | + Text | + Images | + Images + Text | Text-only improvement over covariates |
|---|---:|---:|---:|---:|---:|
| Mean baseline | 4.115 | 4.115 | 4.115 | 4.115 | +0.0% |
| Ridge | 3.096 | 3.096 | 3.090 | 3.090 | +0.0% |
| ElasticNet | 2.705 | 2.705 | 2.704 | 2.705 | -0.0% |
| HistGB | 1.732 | 1.734 | 1.725 | 1.722 | -0.2% |
| RandomForest | 1.815 | 1.819 | 1.758 | 1.759 | -0.2% |

Keyword counts provided little incremental benefit. The release-presence indicator was already in the covariate pipeline. A later fold-fitted TF-IDF/SVD experiment also failed to improve its comparison baseline: MAE 1.7695 versus 1.7362. These findings apply to the tested representations and configurations; they do not prove that the text has no useful information.

### Pooled Versus Category-Specific Modeling

Historical Dataset C + images + HistGB comparison:

| Approach | Combined MAE | Combined RMSE |
|---|---:|---:|
| Pooled universal model | 1.7247 | 2.667 |
| Three category-specific models | 1.735 | Approximately 2.698 |

Separate modeling did not improve this comparison. Pooled modeling retains all three categories in one training table, while each separate model uses about 3,927 rows. The simpler pooled approach was retained.

The project notes also record unequal category errors: approximately 3.09 MAE for `all_drugs`, 1.39 for `all_opioids`, and 0.73 for `all_stimulants`. These are historical diagnostic values, not a new evaluation of the latest candidate.

### Loss, Multitask Training, and Ensemble Weights

| Experiment | Recorded five-fold scored MAE | Development decision |
|---|---:|---|
| Default single HistGB + images | 1.7247 | Single-model reference |
| HistGB/RF equal-weight average | 1.7125 | Ensemble reference |
| Train on all eight categories | 1.7200 | Not retained |
| HistGB with absolute-error loss only | 1.7319 | Not retained by itself |
| Absolute-error HistGB with tuning | 1.7206 | Retained in later candidate |
| TF-IDF text experiment | 1.7695 | Not retained; paired baseline 1.7362 |
| Tuned HistGB/RF weights 0.7/0.3 | 1.7086 | Latest candidate |

The tuning script evaluates out-of-fold predictions for candidate HistGB variants and scans blend weights. The historical notes report the tuned configuration above. The experiments are preserved in scripts 13-15; the candidate implementation is in the submission notebook.

## Submission Notebook

`notebooks/kaggle_submission.ipynb` is the self-contained competition candidate. It:

1. Detects the official data root.
2. Loads labels, covariates, and the prediction template.
3. Calculates MAT image descriptors directly from PNG files.
4. Applies Dataset C weather processing and numeric/categorical preprocessing.
5. Fits the tuned HistGB and RF models on all labeled scored-category rows.
6. Blends nonnegative predictions using weights 0.7 and 0.3.
7. Writes `/kaggle/working/submission.csv` with exactly `row_id` and `rate_per_10000_ed_visits`.

HistGB parameters include `loss="absolute_error"`, `learning_rate=0.05`, `max_iter=600`, `max_leaf_nodes=31`, `l2_regularization=1.0`, and `random_state=42`. RF uses 300 trees, `min_samples_leaf=5`, and `random_state=42`.

The notebook does not depend on saved `outputs/` tables or the development helper file. It reads the current template rather than hard-coding the current holdout's row IDs, so it can adapt to a later evaluation window.

For a local script run, set the official data root and a separate output directory:

```bash
STAIX_DATA=/path/to/official/data STAIX_WORK=/path/to/run-output python notebooks/kaggle_submission.py
```

Required libraries include NumPy, pandas, Pillow, and scikit-learn. The original historical environment was not pinned in this repository.

**Check execution logs:** the original submission implementation catches modeling exceptions and writes constant training-median predictions as a fallback. A CSV file alone does not establish successful model training. Confirm that both model fits completed and that no fallback warning occurred.

## Output Files

### Base Merged Tables

Created by notebook 01:

```text
train_universal_merged.csv
train_all_drugs_merged.csv
train_all_opioids_merged.csv
train_all_stimulants_merged.csv
val_universal_merged.csv
val_all_drugs_merged.csv
val_all_opioids_merged.csv
val_all_stimulants_merged.csv
```

### Fixed-Split Modeling Tables

Pooled tables follow:

```text
universal_A_no_weather_train.csv
universal_A_no_weather_val.csv
universal_B_weather_period_median_train.csv
universal_B_weather_period_median_val.csv
universal_C_weather_similar_state_train.csv
universal_C_weather_similar_state_val.csv
```

The same six suffixes are generated for each prefix `all_drugs`, `all_opioids`, and `all_stimulants`.

### Feature and Comparison Tables

```text
universal_period_model_comparison.csv
category_period_model_comparison.csv
image_features.csv
image_feature_gains.csv
text_features.csv
text_feature_gains.csv
```

## Reproduction Notes

- Keep the original merged tables as inputs to the development experiments.
- Re-run 04, 05, and 08 to regenerate the corresponding comparison results.
- Re-running 02 or 03 overwrites their generated fixed-split tables.
- Use whole-period splits for model assessment; do not substitute random row splits.
- Keep imputation reference statistics within the training fold.
- Distinguish local model-selection scores from official competition evaluation.
- Do not interpret these predictive comparisons as causal evidence about MAT access or public health interventions.
