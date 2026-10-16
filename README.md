# Movie Rating Prediction — Pre-Release IMDb Rating Model

Predicting a movie's IMDb average rating **before release**, using only information that is available prior to screening.

**Team project** by Liel Leone & Gil Hatiel — Machine Learning course, Industrial Engineering & Management (Data Science track), Ariel University.
Original team repository: [gil261/movie-rating-prediction](https://github.com/gil261/movie-rating-prediction)

---

## The Problem

Given basic metadata on ~134K movies (year, genres, runtime, cast, country, language, budget, plot), estimate `averageRating` (1–10).
Columns that are only known after release (`numVotes`, `BoxOffice`) were removed to prevent **data leakage**.

## Data

| | |
|---|---|
| Raw dataset | 133,884 movies × 13 columns (course-provided) |
| After cleaning | 115,560 movies with a valid target × 26 engineered features |
| External sources | IMDb `title.crew.tsv` and `name.basics.tsv` (director and actor names, 300K+ name records matched) |

Data quality issues handled in `prepare_data`:
- Mixed formats in the same column (e.g. `Comedy,Horror` vs. `['Drama', 'Romance']`)
- Heavy missingness: budget 87.6%, country 59.9%, language 58.4%, plot 55.4%
- Invalid values (release year `0`), inconsistent country names (`USA`, `U.S.A.`, `United States of America`)

> Data files are **not included** in this repository. Place `dataset.csv`, `title.crew.tsv` and `name.basics.tsv` in the project root to run the notebook. The IMDb files are available at [developer.imdb.com/non-commercial-datasets](https://developer.imdb.com/non-commercial-datasets/).

## Approach

1. **EDA** — distributions, missing-value profile, format inconsistencies, coverage checks for every engineered feature.
2. **Feature engineering** (`prepare_data`) — deterministic, row-level features only:
   decade, runtime category, 15 genre flags, number of genres, cleaned country, English-language flag, budget-reported flag, plot length, number of directors, acclaimed director/actor flags (lists built from external criteria such as Academy Awards and major festivals, not from the dataset's ratings).
3. **Leakage-safe pipeline** — imputation, scaling and one-hot encoding live inside a scikit-learn `Pipeline` / `ColumnTransformer`, so they are fit on training folds only.
4. **Modeling** — ElasticNet and Random Forest, hyperparameters tuned with `GridSearchCV` (10-fold) on the training split.
5. **Evaluation** — 10-fold cross-validation (RMSE, MAE, R²) and out-of-fold predictions for analysis.
6. **Error analysis** — top 10 over- and under-predictions, feature profiles of the outliers, overlap between the two models.
7. **Fairness analysis** — model error by genre and by release decade.
8. **Interpretability** — ElasticNet coefficients and Random Forest SHAP values.

## Results (10-fold CV)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| ElasticNet (`alpha=0.0001`, `l1_ratio=0.1`) | 1.1317 ± 0.0100 | 0.8665 ± 0.0069 | 0.2336 ± 0.0072 |
| Random Forest (`n_estimators=300`, `max_depth=20`, `min_samples_split=10`) | **1.1129 ± 0.0094** | **0.8463 ± 0.0073** | **0.2588 ± 0.0064** |

For reference, a naive model that always predicts the mean rating has an RMSE of ≈1.29 (the target's standard deviation, see EDA), so Random Forest reduces RMSE by ~14% over this baseline.

## Key Findings

- **Errors concentrate at the extremes.** Ratings are bell-shaped around 6.07; the model regresses toward the center, overpredicting very low-rated films and underpredicting very high-rated niche films.
- **Documentary genre is a double-edged signal.** It is the strongest positive feature in both models, and 9 of the 10 largest overpredictions are documentaries.
- **Errors are driven by the features, not by the algorithm.** The two models share 6/10 of their top overpredictions and 4/10 of their top underpredictions.
- **Performance drops for recent films.** RMSE rises steadily by decade, peaking in the 2020s (1.248, ~12% above the overall average), likely because of sparser metadata and ratings based on few votes.
- **Proposed next features:** language–country match, a non-English/non-Western flag, and a continuous (log-scaled) budget from external sources.

## Repository Structure

| File | Description |
|---|---|
| `movie_rating_prediction.ipynb` | Full pipeline: EDA, `prepare_data`, training, evaluation, error, fairness and feature-importance analysis |
| `model.pkl` | Final ElasticNet pipeline trained on all labeled data (course requirement) |
| `report_hebrew.pdf` | Full written report (Hebrew) |
| `requirements.txt` | Library versions |

## How to Run

```bash
pip install -r requirements.txt
# place dataset.csv, title.crew.tsv, name.basics.tsv in the project root
jupyter notebook movie_rating_prediction.ipynb
```

Using the saved model:

```python
import pandas as pd, joblib
df = pd.read_csv("dataset.csv")
X = prepare_data(df)          # defined in the notebook
model = joblib.load("model.pkl")
y_pred = model.predict(X)
```

Python 3.10+, `random_state=42` throughout.
