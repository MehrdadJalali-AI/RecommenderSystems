from __future__ import annotations

import csv
import json
from itertools import count
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "chapter_04_model_based_collaborative_filtering"
DATA = CHAPTER / "data"


MOVIES = [
    {"movie_id": 1, "title": "Action Hero", "genre": "Action"},
    {"movie_id": 2, "title": "Space Journey", "genre": "Sci-Fi"},
    {"movie_id": 3, "title": "Love Story", "genre": "Romance"},
    {"movie_id": 4, "title": "Funny Days", "genre": "Comedy"},
    {"movie_id": 5, "title": "Mystery Night", "genre": "Thriller"},
    {"movie_id": 6, "title": "Robot Future", "genre": "Sci-Fi"},
]


RATINGS = [
    {"user_id": "Anna", "movie_id": 1, "rating": 5},
    {"user_id": "Anna", "movie_id": 3, "rating": 2},
    {"user_id": "Anna", "movie_id": 4, "rating": 3},
    {"user_id": "Ben", "movie_id": 2, "rating": 5},
    {"user_id": "Ben", "movie_id": 3, "rating": 4},
    {"user_id": "Ben", "movie_id": 4, "rating": 4},
    {"user_id": "Ben", "movie_id": 6, "rating": 5},
    {"user_id": "Sara", "movie_id": 1, "rating": 4},
    {"user_id": "Sara", "movie_id": 3, "rating": 1},
    {"user_id": "Sara", "movie_id": 5, "rating": 4},
    {"user_id": "Sara", "movie_id": 6, "rating": 4},
    {"user_id": "Liam", "movie_id": 1, "rating": 2},
    {"user_id": "Liam", "movie_id": 2, "rating": 4},
    {"user_id": "Liam", "movie_id": 3, "rating": 5},
    {"user_id": "Liam", "movie_id": 4, "rating": 5},
    {"user_id": "Mia", "movie_id": 1, "rating": 5},
    {"user_id": "Mia", "movie_id": 2, "rating": 4},
    {"user_id": "Mia", "movie_id": 5, "rating": 5},
    {"user_id": "Mia", "movie_id": 6, "rating": 4},
    {"user_id": "Omar", "movie_id": 2, "rating": 2},
    {"user_id": "Omar", "movie_id": 3, "rating": 5},
    {"user_id": "Omar", "movie_id": 4, "rating": 4},
    {"user_id": "Omar", "movie_id": 5, "rating": 2},
]


CELL_COUNTER = count(1)


def nb(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": f"cell-{next(CELL_COUNTER):04d}", "metadata": {}, "source": text.strip() + "\n"}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"cell-{next(CELL_COUNTER):04d}",
        "metadata": {},
        "outputs": [],
        "source": text.strip() + "\n",
    }


def challenge_code(number: int) -> dict:
    return code(f"""
# Challenge {number}
# Modify or extend the code as described above.

# Write your code below:



""")


COMMON_LOAD = r"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import NMF
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIRS = [
    Path("data"),
    Path("../data"),
    Path("chapter_04_model_based_collaborative_filtering/data"),
]
GITHUB_DATA_URL = "https://raw.githubusercontent.com/MehrdadJalali-AI/RecommenderSystems/main/chapter_04_model_based_collaborative_filtering/data"

def read_chapter4_csv(filename):
    for data_dir in DATA_DIRS:
        csv_path = data_dir / filename
        if csv_path.exists():
            print(f"Loaded {filename} from {csv_path}")
            return pd.read_csv(csv_path)
    url = f"{GITHUB_DATA_URL}/{filename}"
    print(f"Local file not found. Loading {filename} from GitHub raw URL.")
    return pd.read_csv(url)

ratings = read_chapter4_csv("ratings_chapter4.csv")
movies = read_chapter4_csv("movies_chapter4.csv")
ratings_named = ratings.merge(movies, on="movie_id", how="left")
rating_matrix = ratings_named.pivot_table(index="user_id", columns="title", values="rating")
rating_matrix
"""


PACKAGE_NOTE = """
## Packages and key functions used

- `pandas` is used for tables: `read_csv`, `merge`, `pivot_table`, `groupby`, `sort_values`, and `dropna`.
- `numpy` is used for matrix operations: `dot`, `linalg.svd`, `sqrt`, `clip`, random initialization, and array indexing.
- `sklearn.decomposition.NMF` fits a non-negative matrix factorization model.
- `sklearn.metrics.mean_absolute_error` and `mean_squared_error` evaluate rating prediction error.
- `Path` helps the notebook find CSV files locally or load them from GitHub when opened in Colab.
"""


def write_data() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "movies_chapter4.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(MOVIES[0]))
        writer.writeheader()
        writer.writerows(MOVIES)
    with (DATA / "ratings_chapter4.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(RATINGS[0]))
        writer.writeheader()
        writer.writerows(RATINGS)


def write_notebook(name: str, cells: list[dict]) -> None:
    CHAPTER.mkdir(parents=True, exist_ok=True)
    (CHAPTER / name).write_text(json.dumps(nb(cells), indent=2), encoding="utf-8")


def chapter4_notebook() -> list[dict]:
    return [
        md("""
# Chapter 4 Practical: Model-Based Collaborative Filtering

This notebook follows `RS_C4_V4.pdf`: Collaborative Filtering - Model-Based.

Learning objectives:
- Understand how model-based CF differs from memory-based CF.
- Represent users and items with latent factors.
- Use matrix factorization to predict missing ratings.
- Compare simple SVD, ALS, and NMF examples.
- Evaluate rating predictions with MAE and RMSE.
"""),
        code(COMMON_LOAD),
        md(PACKAGE_NOTE),
        md("""
## Part 1 - From missing ratings to a learned model

Model-based collaborative filtering does not search for nearest neighbors every time. Instead, it learns compact user and item profiles from the known ratings.

In the matrix above:
- rows are users,
- columns are movies,
- numbers are observed ratings,
- missing cells are unknown preferences that we want to predict.
"""),
        code(r"""
n_users, n_items = rating_matrix.shape
n_known = rating_matrix.notna().sum().sum()
density = n_known / (n_users * n_items)

pd.DataFrame({
    "measure": ["users", "items", "known ratings", "possible ratings", "density", "sparsity"],
    "value": [n_users, n_items, n_known, n_users * n_items, round(density, 3), round(1 - density, 3)],
})
"""),
        md("""
## Part 2 - Latent factors and dot products

Latent factors are hidden preference dimensions learned from ratings. In the lecture, possible interpretations include Action, Romance, Classic, Modern, Popular, or Specialized.

The names below are only for teaching. In real matrix factorization, the model learns numerical factors automatically and they may not have clear human names.

The dot product estimates a match score:

`score(user, item) = user_factors dot item_factors`
"""),
        code(r"""
manual_user_factors = pd.DataFrame(
    [[0.9, 0.2], [0.2, 0.9]],
    index=["Anna", "Ben"],
    columns=["Action factor", "Romance factor"],
)

manual_item_factors = pd.DataFrame(
    [[0.8, 0.1], [0.1, 0.8], [0.7, 0.3]],
    index=["Action Hero", "Love Story", "Space Journey"],
    columns=["Action factor", "Romance factor"],
)

match_scores = manual_user_factors.dot(manual_item_factors.T)
match_scores.round(2)
"""),
        md("""
## Part 3 - SVD-style matrix factorization

Singular Value Decomposition (SVD) decomposes a matrix into compact latent dimensions. A full recommender would learn from only observed ratings. For this small classroom example, we first fill missing values with user means so `np.linalg.svd()` can work on a complete matrix.

Important functions:
- `fillna()` creates a complete teaching matrix.
- `np.linalg.svd()` factorizes the matrix.
- `k_factors` controls how many latent dimensions are kept.
- `np.clip()` keeps predictions inside the 1-5 rating range.
"""),
        code(r"""
filled_matrix = rating_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)
matrix_values = filled_matrix.values

U, singular_values, Vt = np.linalg.svd(matrix_values, full_matrices=False)

k_factors = 2
U_k = U[:, :k_factors]
S_k = np.diag(singular_values[:k_factors])
Vt_k = Vt[:k_factors, :]

svd_prediction_values = U_k @ S_k @ Vt_k
svd_predictions = pd.DataFrame(
    np.clip(svd_prediction_values, 1, 5),
    index=rating_matrix.index,
    columns=rating_matrix.columns,
)

svd_predictions.round(2)
"""),
        md("""
The next cell recommends unseen movies for one target user by ranking the predicted ratings for missing entries only.
"""),
        code(r"""
target_user = "Anna"
seen_items = rating_matrix.loc[target_user].dropna().index
svd_recommendations = (
    svd_predictions.loc[target_user]
    .drop(labels=seen_items)
    .sort_values(ascending=False)
    .rename("predicted_rating")
    .to_frame()
)
svd_recommendations.round(2)
"""),
        md("""
## Part 4 - ALS from observed ratings

Alternating Least Squares (ALS) alternates between two steps:

1. keep item factors fixed and update user factors,
2. keep user factors fixed and update item factors.

This simple implementation uses only observed ratings. It is small enough for students to inspect, but it mirrors the main lecture idea.
"""),
        code(r"""
def fit_als(matrix, n_factors=2, n_iterations=20, regularization=0.1, random_state=42):
    rng = np.random.default_rng(random_state)
    rating_values = matrix.values.astype(float)
    observed = ~np.isnan(rating_values)
    n_users, n_items = rating_values.shape

    user_factors = rng.normal(0, 0.1, size=(n_users, n_factors))
    item_factors = rng.normal(0, 0.1, size=(n_items, n_factors))
    identity = np.eye(n_factors)

    for _ in range(n_iterations):
        for u in range(n_users):
            item_ids = np.where(observed[u])[0]
            if len(item_ids) == 0:
                continue
            V = item_factors[item_ids]
            r = rating_values[u, item_ids]
            user_factors[u] = np.linalg.solve(V.T @ V + regularization * identity, V.T @ r)

        for i in range(n_items):
            user_ids = np.where(observed[:, i])[0]
            if len(user_ids) == 0:
                continue
            U_obs = user_factors[user_ids]
            r = rating_values[user_ids, i]
            item_factors[i] = np.linalg.solve(U_obs.T @ U_obs + regularization * identity, U_obs.T @ r)

    predictions = user_factors @ item_factors.T
    predictions = np.clip(predictions, 1, 5)
    return user_factors, item_factors, pd.DataFrame(predictions, index=matrix.index, columns=matrix.columns)

als_user_factors, als_item_factors, als_predictions = fit_als(rating_matrix, n_factors=2)
als_predictions.round(2)
"""),
        md("""
The factor matrices are the learned user and item profiles. They are reusable: once learned, the recommender can predict many missing ratings quickly.
"""),
        code(r"""
user_factor_table = pd.DataFrame(
    als_user_factors,
    index=rating_matrix.index,
    columns=["latent_factor_1", "latent_factor_2"],
)

item_factor_table = pd.DataFrame(
    als_item_factors,
    index=rating_matrix.columns,
    columns=["latent_factor_1", "latent_factor_2"],
)

print("User latent factors:")
display(user_factor_table.round(2))
print("Item latent factors:")
display(item_factor_table.round(2))
"""),
        md("""
## Part 5 - NMF with non-negative factors

Non-Negative Matrix Factorization (NMF) learns only zero or positive factor values. This can make factors easier to interpret because profiles are combined additively.

`sklearn.decomposition.NMF` needs a complete non-negative matrix. For this classroom example, missing values are filled with item means. This filling is only a teaching step, not a claim that missing ratings are real ratings.
"""),
        code(r"""
nmf_input = rating_matrix.apply(lambda col: col.fillna(col.mean()), axis=0)

nmf_model = NMF(n_components=2, init="random", random_state=7, max_iter=1000)
nmf_user_factors = nmf_model.fit_transform(nmf_input)
nmf_item_factors = nmf_model.components_

nmf_prediction_values = nmf_user_factors @ nmf_item_factors
nmf_predictions = pd.DataFrame(
    np.clip(nmf_prediction_values, 1, 5),
    index=rating_matrix.index,
    columns=rating_matrix.columns,
)

nmf_predictions.round(2)
"""),
        md("""
## Part 6 - Choosing k with validation error

The number of latent factors `k` controls model complexity.

- small `k`: simple model, may underfit,
- large `k`: flexible model, may overfit,
- balanced `k`: chosen using validation error such as RMSE.

The next example hides one rating per user, trains ALS on the remaining ratings, and evaluates MAE/RMSE on the hidden ratings.
"""),
        code(r"""
test_rows = ratings_named.groupby("user_id", group_keys=False).sample(n=1, random_state=11)
train_rows = ratings_named.drop(test_rows.index)
train_matrix = train_rows.pivot_table(index="user_id", columns="title", values="rating")
train_matrix = train_matrix.reindex(index=rating_matrix.index, columns=rating_matrix.columns)

results = []
for k in [1, 2, 3]:
    _, _, train_predictions = fit_als(train_matrix, n_factors=k, n_iterations=30, regularization=0.2, random_state=7)
    actual = []
    predicted = []
    for _, row in test_rows.iterrows():
        user = row["user_id"]
        title = row["title"]
        actual.append(row["rating"])
        predicted.append(train_predictions.loc[user, title])
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    results.append({"k_factors": k, "MAE": mae, "RMSE": rmse})

pd.DataFrame(results).round(3)
"""),
        md("""
# Challenges

### Challenge 1 - Change the number of latent factors

**Goal:**
Investigate how `k_factors` changes SVD predictions and recommendations.

**What to do:**

1. In the SVD section, change `k_factors = 2` to `k_factors = 1`.
2. Rerun the SVD prediction and recommendation cells.
3. Then try `k_factors = 3`.
4. Compare Anna's recommendation list for the different values of `k_factors`.
"""),
        challenge_code(1),
        md("""
### Your observations

Which value of `k_factors` changed the recommendations most? Did a larger `k` always look better? Why can too small or too large `k` be a problem?

> Write your observations here:
>
> ................................................................................
>
> ................................................................................
>
> ................................................................................
"""),
        md("""
### Challenge 2 - Compare ALS and NMF predictions

**Goal:**
Compare two model-based approaches on the same target user.

**What to do:**

1. Choose one `target_user`, such as `"Anna"` or `"Ben"`.
2. Rank unseen movies for that user using `als_predictions`.
3. Rank unseen movies for the same user using `nmf_predictions`.
4. Compare whether ALS and NMF recommend the same top movie.
"""),
        challenge_code(2),
        md("""
### Your observations

Did ALS and NMF choose the same top recommendation? Which prediction table looked easier to interpret? Why might different factorization methods produce different rankings?

> Write your observations here:
>
> ................................................................................
>
> ................................................................................
>
> ................................................................................
"""),
        md("""
### Challenge 3 - Concept Check: Model-Based CF

This challenge requires no programming.

Memory-based CF finds similar users or items directly from the rating matrix. Model-based CF learns latent user and item profiles first.

Explain in your own words:

1. Why can learned latent factors help when the rating matrix is sparse?
2. Why do we need validation error when choosing the number of latent factors?
3. Why might NMF factors be easier to interpret than factors with negative values?

### Your explanation

> ................................................................................
>
> ................................................................................
>
> ................................................................................
>
> ................................................................................
"""),
    ]


def write_readme() -> None:
    (CHAPTER / "README.md").write_text(
        """# Chapter 4: Collaborative Filtering - Model-Based

This folder contains a compact practical notebook for Chapter 4. It follows the revised lecture topics in `RS_C4_V4.pdf`.

## Notebook Path

| Order | Notebook | Main idea |
| --- | --- | --- |
| 01 | `01_model_based_collaborative_filtering.ipynb` | Latent factors, matrix factorization, SVD, ALS, NMF, k selection, MAE/RMSE |

## Data

The `data/` folder contains a small classroom movie-rating dataset:

- `movies_chapter4.csv`
- `ratings_chapter4.csv`

The notebook is intentionally small so students can inspect intermediate factor matrices and prediction tables.
""",
        encoding="utf-8",
    )


def build() -> None:
    write_data()
    write_notebook("01_model_based_collaborative_filtering.ipynb", chapter4_notebook())
    write_readme()


if __name__ == "__main__":
    build()
