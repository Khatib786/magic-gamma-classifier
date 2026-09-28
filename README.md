# MAGIC Gamma Telescope - Physics-Informed Gamma/Hadron Separation

This project separates gamma-ray air showers from the much more common hadronic cosmic-ray background using Hillas image parameters from the [MAGIC](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope) Imaging Atmospheric Cherenkov Telescope (IACT) dataset from UCI.

The project is not just a simple classifier. The complete pipeline follows several checks that are important in a physics analysis.

* **No data leakage.** Feature engineering is included inside the scikit-learn `Pipeline`. This means derived features are fitted only using the training fold and are never fitted using data that will later be used for evaluation.

* **Using a physics-relevant metric.** In addition to ROC-AUC, the pipeline calculates the **Q-factor** (signal efficiency / √background efficiency). This is a common figure of merit in Imaging Atmospheric Cherenkov Telescope (IACT) analyses and is measured at a fixed background-acceptance rate.

* **Statistical comparison.** The effect of the hand-engineered physics features is tested using a **paired t-test across cross-validation folds** for each model, rather than comparing only two final numbers.

* **No silent replacement with fake data.** If the real UCI dataset cannot be downloaded, the run stops with a `RuntimeError`. It does not silently replace the dataset with synthetic values that could be mistaken for real experimental results.

* **Test coverage.** The `features`, `metrics`, `data`, and `models` modules have unit tests using synthetic fixtures. These tests check the code itself rather than producing physics results. See [tests/](tests/).

## Project Layout

```text
magic-gamma-classifier/

├── src/magic_classifier/
│   ├── config.py         # constants, paths, random seed
│   ├── data.py           # UCI loading + local-file/URL fallback, no synthetic data
│   ├── features.py       # HillasFeatureBuilder — the physics feature engineering
│   ├── metrics.py        # ROC-AUC, Q-factor, threshold-at-FPR
│   ├── models.py         # estimators, Pipeline factory, CV, paired t-test, Optuna
│   ├── visualize.py      # all figure generation
│   ├── pipeline.py       # end-to-end experiment + CLI
│   └── logging_utils.py  # console + rotating file logging

├── scripts/
│   └── make_paper_tables.py  

├── tests/                    # pytest unit tests (synthetic fixtures only)
├── run.py                    # `python run.py` convenience entry point
├── output/figures
├── requirements.txt
├── pyproject.toml
├── README.md
└── paper.md                  # write-up in the shape of a short physics/ML paper
```

## Setup

```bash
git clone <this-repo-url>

cd magic-gamma-classifier

python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

LightGBM, Optuna, and `ucimlrepo` are optional. They are listed in `requirements.txt`, but the pipeline can continue if one of them is not installed. In that case, it logs a warning and skips the related part.

## Data

The loader in `src/magic_classifier/data.py` looks for the dataset in the following order:

1. `--data <path>` if provided
2. A local `magic04.data` / `magic04.csv` in the working directory or `data/`
3. The [`ucimlrepo`](https://pypi.org/project/ucimlrepo/) package using `fetch_ucirepo(id=159)`
4. A direct download from the UCI archive

If none of these options works, the run raises a `RuntimeError`. It does not fall back to fake data. Therefore, any results printed by the pipeline come from the real dataset.

The dataset can also be downloaded manually:

```bash
mkdir -p data

curl -o data/magic04.data \
https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data
```

## Running

Run the complete experiment with:

```bash
python run.py
```

This runs the 5-fold cross-validation, hold-out evaluation, and figure generation.

To also run the Optuna-tuned LightGBM model:

```bash
python run.py --tune --trials 30
```

To use 10-fold cross-validation and a 20% test set:

```bash
python run.py --cv 10 --test-size 0.2
```

To skip the slower t-SNE embedding:

```bash
python run.py --no-embedding
```

The project can also be installed as a console script:

```bash
pip install -e .

magic-classifier --tune
```

Outputs are saved in `outputs/`. This can be changed with `--out`.

| File                        | Contents                                                          |
| --------------------------- | ----------------------------------------------------------------- |
| `dataset.csv`               | The loaded and normalised dataset (`target` = 1 gamma / 0 hadron) |
| `results.csv`               | Hold-out AUC / Q-factor / TPR per model × feature-set             |
| `feature_significance.json` | Paired t-test of engineered vs. raw features, per model           |
| `02_physics_features.png`   | Engineered-feature histograms for gamma and hadron events         |
| `03_roc_curves.png`         | ROC curves for all model/feature combinations                     |
| `04_summary.png`            | Grouped-bar comparison of AUC and Q-factor                        |
| `05_feature_importance.png` | Random-forest feature importances                                 |
| `06_feature_embedding.png`  | t-SNE of raw vs. engineered feature space                         |

Logs are written to both the console and `logs/experiment.log`. Log rotation is set to 5 MB with 3 backups.

## Reproducing the Paper's Tables

Run the main experiment first:

```bash
python run.py
```

Then generate the tables used in `paper.md`:

```bash
python scripts/make_paper_tables.py
```

The tables are generated from the results stored in `outputs/`.

## Testing

Install the development requirements:

```bash
pip install -r requirements-dev.txt
```

Then run the tests:

```bash
pytest -q
```

The tests use a small synthetic fixture from `tests/conftest.py` that follows the structure of the real dataset. They are intended to test the code, not to produce or support physics conclusions.

All physics results and conclusions come from the real UCI dataset when running:

```bash
python run.py
```

## Engineered Features

The derived features are created by `HillasFeatureBuilder` and are fitted only on the training folds.

| Feature      | Definition                   | Physics motivation                        |
| ------------ | ---------------------------- | ----------------------------------------- |
| `elongation` | `fLength / fWidth`           | Gammas produce more elongated ellipses    |
| `asym_ratio` | `\|fAsym\| / fLength`        | Hadron showers are more asymmetric        |
| `conc_ratio` | `fConc1 / fConc`             | Light-concentration profile               |
| `alpha_size` | `fAlpha / \|fSize\|`         | Orientation angle normalised by intensity |
| `area_proxy` | `π · fLength · fWidth`       | Ellipse area, a shower-size proxy         |
| `m3_ratio`   | `\|fM3Long\| / \|fM3Trans\|` | Third-moment asymmetry ratio              |

## License

MIT — see [LICENSE](LICENSE).
