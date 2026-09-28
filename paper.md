# Physics-Informed Gamma/Hadron Separation on the MAGIC Gamma Telescope Dataset

**Author:** Khatib Kabir

## Abstract

Imaging Atmospheric Cherenkov Telescopes (IACTs) such as MAGIC detect many more hadronic cosmic-ray showers than genuine gamma-ray events. Separating these two classes is therefore an important classification problem, where false positives can reduce the quality of the astrophysical signal.

This study uses the UCI MAGIC Gamma Telescope dataset, which contains 19,020 events described by 10 Hillas image parameters. The analysis pipeline is designed around several checks commonly used in physics analyses. These include leakage-free feature engineering, the Q-factor alongside ROC-AUC, and a paired statistical test to determine whether the additional physics-based features provide a measurable improvement.

Five model families are compared: logistic regression, gradient boosting, random forest, MLP, and, when the optional dependencies are available, LightGBM with early stopping and Optuna-tuned hyperparameters. Each model is evaluated using the original Hillas parameters both with and without six physics-motivated derived features.

## 1. Introduction

The MAGIC telescope records the Cherenkov light produced when high-energy particles create air showers in the atmosphere. The resulting shower image can be described using **Hillas parameters**, including length, width, size, concentration, asymmetry, third moments, the angle to the source (`alpha`), and distance.

Gamma-ray showers generally produce more compact and elongated images and tend to be better aligned with the source direction than hadronic showers. However, the distributions of the two classes overlap considerably. This makes the problem suitable for machine-learning classification rather than relying only on simple geometric cuts.

The dataset contains 19,020 simulated events with 10 continuous Hillas features and a binary label:

* `g` = gamma / signal
* `h` = hadron / background

The dataset contains approximately 65% gamma events and 35% hadron events.

## 2. Method

### 2.1 Data and Leakage Control

All feature engineering in `src/magic_classifier/features.py` is implemented as a scikit-learn `TransformerMixin` and placed inside the modeling `Pipeline`, together with standardisation.

This keeps the feature-processing steps within the same train/test and cross-validation structure as the models. Each fold therefore processes its data without using information from the data that will later be used for evaluation.

In the current implementation, the engineered features are row-wise ratios and products, so they do not have fitted statistics that could leak between folds. Keeping the transformation inside the pipeline also allows the same structure to be used for features that require fold-specific fitting.

### 2.2 Engineered Features

Six additional features are created from the existing Hillas parameters.

| Feature      | Definition                   | Physics motivation                        |
| ------------ | ---------------------------- | ----------------------------------------- |
| `elongation` | `fLength / fWidth`           | Gammas produce more elongated ellipses    |
| `asym_ratio` | `\|fAsym\| / fLength`        | Hadron showers are more asymmetric        |
| `conc_ratio` | `fConc1 / fConc`             | Light-concentration profile               |
| `alpha_size` | `fAlpha / \|fSize\|`         | Orientation angle normalised by intensity |
| `area_proxy` | `π · fLength · fWidth`       | Ellipse area, a shower-size proxy         |
| `m3_ratio`   | `\|fM3Long\| / \|fM3Trans\|` | Third-moment asymmetry ratio              |

### 2.3 Models

The baseline model set contains logistic regression, gradient boosting, random forest, and an MLP. These models are implemented in `src/magic_classifier/models.py`.

LightGBM is also supported when its optional dependency is installed. The pipeline includes an inner-validation early-stopping wrapper for LightGBM, as well as an Optuna-tuned LightGBM variant.

### 2.4 Metrics

The models are evaluated using three main measures.

* **ROC-AUC** — measures the overall ranking performance of the classifier.

* **Q-factor** — defined as `Q = ε_signal / √ε_background`. This is a standard figure of merit in IACT analyses. It is reported at a fixed 1% background-acceptance operating point and also at the best value obtained across thresholds.

* **TPR at fixed FPR = 0.01** — measures the signal efficiency retained when the background acceptance is fixed at 1%.

### 2.5 Do the Engineered Features Actually Help?

The comparison does not rely on a single AUC value from each feature set.

The `compare_features` procedure runs stratified k-fold cross-validation for both the raw and engineered feature configurations. The folds are kept aligned, and the AUC difference between the two configurations is calculated for each fold.

A **paired t-test** using `scipy.stats.ttest_rel` is then applied to these per-fold AUC differences for each model.

An improvement is considered statistically significant when `p < 0.05`.

This provides a way to distinguish a consistent improvement across folds from a higher AUC that may simply result from variation between individual runs or splits.

## 3. Results

The tables in this section are generated from the actual experiment outputs. The script `scripts/make_paper_tables.py` reads `outputs/results.csv` and `outputs/feature_significance.json` and fills the tables after a real experiment has been run.

Until the experiment is run, the tables remain as placeholders.

### 3.1 Hold-out Performance

<!-- RESULTS_TABLE_START -->

*(Run `python run.py && python scripts/make_paper_tables.py` to fill this table.)*

<!-- RESULTS_TABLE_END -->

### 3.2 Do the Engineered Features Help? — Paired t-test Across CV Folds

<!-- SIGNIFICANCE_TABLE_START -->

*(Run `python run.py && python scripts/make_paper_tables.py` to fill this table.)*

<!-- SIGNIFICANCE_TABLE_END -->

### 3.3 Figures

The main figures generated by the pipeline are:

* `outputs/02_physics_features.png` — distributions of the engineered features for gamma and hadron events
* `outputs/03_roc_curves.png` — ROC curves for every model and feature-set combination
* `outputs/04_summary.png` — grouped comparison of AUC and Q-factor
* `outputs/05_feature_importance.png` — random-forest feature importances, with engineered features highlighted
* `outputs/06_feature_embedding.png` — t-SNE projection of the raw and engineered feature spaces

## 4. Discussion

The final discussion should be completed after the results tables have been generated from a real run.

The main points to examine are:

* Which model gives the highest Q-factor at the realistic 1% background-acceptance point, and how this compares with the model having the highest raw ROC-AUC.

* Which models show a **statistically significant** improvement (`p < 0.05`) from the engineered features, and which models show a numerically higher AUC without enough evidence from the paired t-test.

* Whether the feature-importance results in `05_feature_importance.png` place the engineered ratios among the important features, and whether this agrees with the models that show a significant improvement.

## 5. Limitations

There are several limitations to consider when interpreting the results.

* The dataset consists of Monte Carlo-simulated events rather than measurements from real telescope observations. The Q-factor values should therefore be treated as method comparisons and not as calibrated physical acceptance rates for an operating telescope.

* The engineered features are simple closed-form ratios of the existing Hillas parameters. They do not add new information beyond the original parameters. Their value comes from whether a particular model can use the transformed representation more effectively than the original inputs.

* The Q-factor at a fixed 1% FPR is estimated using a single hold-out split. With approximately 4,700 hold-out events, representing 25% of the 19,020-event dataset, the expected number of background events contributing at that operating point is roughly:

  `4,700 × 0.35 × 0.01 ≈ 16 events`

  Therefore, the Q-factor at exactly 1% background acceptance has meaningful sampling variation and should be considered together with the best achievable Q-factor rather than interpreted on its own.

## 6. Reproducibility

Install the required packages and run the experiment with:

```bash
pip install -r requirements.txt

python run.py --tune          # LightGBM + Optuna tuning, if installed

python scripts/make_paper_tables.py

pytest -q                     # unit tests using synthetic fixtures
```

The random seed is fixed at `RS = 42` in `src/magic_classifier/config.py`. It is used for the train/test split, cross-validation folds, and stochastic estimators.

This fixed seed helps keep the experiment reproducible when it is run again under the same setup.

## References

* R. K. Bock et al., “Methods for multidimensional event classification: a case study using images from a Cherenkov gamma-ray telescope,” *Nuclear Instruments and Methods in Physics Research A*, vol. 516, 2004, pp. 511–528.

* UCI Machine Learning Repository, “MAGIC Gamma Telescope,” dataset 159.
  https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope