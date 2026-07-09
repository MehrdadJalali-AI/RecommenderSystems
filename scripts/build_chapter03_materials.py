from __future__ import annotations

import csv
import json
from itertools import count
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "chapter_03_collaborative_filtering"
DATA = CHAPTER / "data"
DEMOS = CHAPTER / "html_demos"


MOVIES = [
    {"movie_id": 1, "title": "Star Wars", "genre": "Sci-Fi", "year": 1977},
    {"movie_id": 2, "title": "Jurassic Park", "genre": "Adventure", "year": 1993},
    {"movie_id": 3, "title": "Terminator 2", "genre": "Action", "year": 1991},
    {"movie_id": 4, "title": "Independence Day", "genre": "Sci-Fi", "year": 1996},
    {"movie_id": 5, "title": "The Matrix", "genre": "Sci-Fi", "year": 1999},
    {"movie_id": 6, "title": "Toy Story", "genre": "Animation", "year": 1995},
    {"movie_id": 7, "title": "Titanic", "genre": "Romance", "year": 1997},
    {"movie_id": 8, "title": "The Notebook", "genre": "Romance", "year": 2004},
    {"movie_id": 9, "title": "Blade Runner", "genre": "Sci-Fi", "year": 1982},
    {"movie_id": 10, "title": "Finding Nemo", "genre": "Animation", "year": 2003},
]


RATINGS = [
    {"user_id": "Sally", "movie_id": 1, "rating": 7, "days_ago": 420},
    {"user_id": "Sally", "movie_id": 2, "rating": 6, "days_ago": 250},
    {"user_id": "Sally", "movie_id": 3, "rating": 3, "days_ago": 120},
    {"user_id": "Sally", "movie_id": 4, "rating": 7, "days_ago": 20},
    {"user_id": "Sally", "movie_id": 5, "rating": 6, "days_ago": 14},
    {"user_id": "Bob", "movie_id": 1, "rating": 7, "days_ago": 360},
    {"user_id": "Bob", "movie_id": 2, "rating": 4, "days_ago": 270},
    {"user_id": "Bob", "movie_id": 3, "rating": 4, "days_ago": 160},
    {"user_id": "Bob", "movie_id": 4, "rating": 6, "days_ago": 30},
    {"user_id": "Bob", "movie_id": 5, "rating": 7, "days_ago": 6},
    {"user_id": "Chris", "movie_id": 1, "rating": 3, "days_ago": 330},
    {"user_id": "Chris", "movie_id": 2, "rating": 7, "days_ago": 210},
    {"user_id": "Chris", "movie_id": 3, "rating": 7, "days_ago": 90},
    {"user_id": "Chris", "movie_id": 4, "rating": 2, "days_ago": 35},
    {"user_id": "Chris", "movie_id": 6, "rating": 5, "days_ago": 22},
    {"user_id": "Lynn", "movie_id": 1, "rating": 4, "days_ago": 300},
    {"user_id": "Lynn", "movie_id": 2, "rating": 4, "days_ago": 180},
    {"user_id": "Lynn", "movie_id": 3, "rating": 6, "days_ago": 70},
    {"user_id": "Lynn", "movie_id": 4, "rating": 2, "days_ago": 12},
    {"user_id": "Lynn", "movie_id": 6, "rating": 6, "days_ago": 10},
    {"user_id": "Karen", "movie_id": 1, "rating": 7, "days_ago": 280},
    {"user_id": "Karen", "movie_id": 2, "rating": 4, "days_ago": 150},
    {"user_id": "Karen", "movie_id": 3, "rating": 3, "days_ago": 65},
    {"user_id": "Karen", "movie_id": 5, "rating": 6, "days_ago": 8},
    {"user_id": "Alice", "movie_id": 5, "rating": 5, "days_ago": 40},
    {"user_id": "Alice", "movie_id": 9, "rating": 5, "days_ago": 12},
    {"user_id": "Alice", "movie_id": 1, "rating": 4, "days_ago": 180},
    {"user_id": "Alice", "movie_id": 4, "rating": 4, "days_ago": 21},
    {"user_id": "Nina", "movie_id": 6, "rating": 5, "days_ago": 30},
    {"user_id": "Nina", "movie_id": 10, "rating": 5, "days_ago": 7},
    {"user_id": "Nina", "movie_id": 2, "rating": 4, "days_ago": 110},
    {"user_id": "Nina", "movie_id": 7, "rating": 2, "days_ago": 260},
    {"user_id": "Omar", "movie_id": 7, "rating": 5, "days_ago": 15},
    {"user_id": "Omar", "movie_id": 8, "rating": 5, "days_ago": 6},
    {"user_id": "Omar", "movie_id": 2, "rating": 3, "days_ago": 140},
    {"user_id": "Omar", "movie_id": 10, "rating": 2, "days_ago": 90},
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

DATA_DIRS = [
    Path("data"),
    Path("../data"),
    Path("chapter_03_collaborative_filtering/data"),
]
GITHUB_DATA_URL = "https://raw.githubusercontent.com/MehrdadJalali-AI/RecommenderSystems/main/chapter_03_collaborative_filtering/data"

def read_chapter3_csv(filename):
    for data_dir in DATA_DIRS:
        csv_path = data_dir / filename
        if csv_path.exists():
            print(f"Loaded {filename} from {csv_path}")
            return pd.read_csv(csv_path)
    url = f"{GITHUB_DATA_URL}/{filename}"
    print(f"Local file not found. Loading {filename} from GitHub raw URL.")
    return pd.read_csv(url)

ratings = read_chapter3_csv("ratings_chapter3.csv")
movies = read_chapter3_csv("movies_chapter3.csv")
ratings_named = ratings.merge(movies, on="movie_id", how="left")
rating_matrix = ratings_named.pivot_table(index="user_id", columns="title", values="rating")
rating_matrix
"""


PACKAGE_NOTE = """
## Packages and key functions used

- `pandas` is used for tabular data. Important functions in this notebook include `read_csv`, `merge`, `pivot_table`, `groupby`, `agg`, `sort_values`, and `dropna`.
- `numpy` is used for numerical operations such as vector norms, dot products, averages, absolute values, and exponential time-decay weights.
- `pathlib.Path` helps the notebook find local CSV files in Colab, Jupyter, or the repository folder.
- The shared helper `read_chapter3_csv()` first looks for local data files and then falls back to the GitHub raw URL when students open the notebook directly online.
- `rating_matrix` is the central memory-based collaborative filtering object: rows are users, columns are movies, observed numbers are ratings, and missing cells mean unknown preferences.
"""


def write_notebook(name: str, cells: list[dict], folder: Path = CHAPTER) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(nb(cells), indent=2), encoding="utf-8")


def write_data() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    DEMOS.mkdir(parents=True, exist_ok=True)
    with (DATA / "movies_chapter3.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(MOVIES[0]))
        writer.writeheader()
        writer.writerows(MOVIES)
    with (DATA / "ratings_chapter3.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(RATINGS[0]))
        writer.writeheader()
        writer.writerows(RATINGS)


def notebook_01() -> list[dict]:
    return [
        md("""
# Chapter 3 Practical 01: User-Item Matrix and Sparsity

Learning objectives:
- Build a user-item rating matrix from interactions.
- Measure density and sparsity.
- Understand why collaborative filtering needs overlap.
- Visualize rating distributions and missing values.

Slide connection: collaborative filtering motivation, memory-based CF overview, and user-item rating matrix.
"""),
        code(COMMON_LOAD),
        md(PACKAGE_NOTE),
        md("""
## Technique: rating matrix, density, and sparsity

Memory-based collaborative filtering keeps the observed interaction matrix and uses it directly at recommendation time. In this code:

- `rating_matrix.shape` returns the number of users and items.
- `rating_matrix.notna()` marks observed ratings as `True` and missing ratings as `False`.
- `sum().sum()` counts all observed ratings across the whole matrix.
- Density is the fraction of known ratings; sparsity is `1 - density`.

The output table should help students see that collaborative filtering usually works with many missing values.
"""),
        code(r"""
n_users, n_items = rating_matrix.shape
n_known = rating_matrix.notna().sum().sum()
density = n_known / (n_users * n_items)

summary = pd.DataFrame({
    "measure": ["users", "items", "known ratings", "possible ratings", "density", "sparsity"],
    "value": [n_users, n_items, n_known, n_users * n_items, round(density, 3), round(1 - density, 3)],
})
summary
"""),
        md("""
## Technique: visualizing missing values

`matplotlib.pyplot` is used only for a simple heatmap. The function `imshow()` displays a grid where green cells mean a known rating and blank cells mean a missing value. This plot is useful because students can see sparsity before calculating it.
"""),
        code(r"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
ax.imshow(rating_matrix.notna(), aspect="auto", cmap="Greens")
ax.set_xticks(range(len(rating_matrix.columns)), rating_matrix.columns, rotation=45, ha="right")
ax.set_yticks(range(len(rating_matrix.index)), rating_matrix.index)
ax.set_title("Known ratings in the user-item matrix")
ax.set_xlabel("Movie")
ax.set_ylabel("User")
fig.tight_layout()
fig
"""),
        md("""
## Technique: co-rated-item overlap

The overlap table counts how many movies each pair of users both rated. This matters because cosine and Pearson similarity are more reliable when they are based on several shared ratings, not only one accidental match.
"""),
        code(r"""
users = rating_matrix.index
overlap = pd.DataFrame(index=users, columns=users, dtype=int)
for u in users:
    for v in users:
        overlap.loc[u, v] = rating_matrix.loc[[u, v]].notna().all(axis=0).sum()
overlap
"""),
        code(r"""
ratings_named.groupby("title")["rating"].agg(["count", "mean"]).sort_values(["count", "mean"], ascending=False)
"""),
        md("""
# Challenges

### Challenge 1 — Add a Sparse New User

**Goal:**
Investigate how a user with very few ratings changes the user-item matrix and co-rated-item overlap.

**What to do:**

1. Add a new user with only one movie rating to `ratings_named`.
2. Rebuild `rating_matrix` from the updated data.
3. Rerun the sparsity and overlap cells.
4. Compare the new user's overlap with the existing users.
"""),
        challenge_code(1),
        md("""
### Your observations

What happened to matrix sparsity? How much overlap did the new user have with other users? Why does this make memory-based CF difficult?

> Write your observations here:
>
> ................................................................................
>
> ................................................................................
>
> ................................................................................
"""),
        md("""
### Challenge 2 — Add a Rating for a Rare Movie

**Goal:**
Investigate how adding one extra interaction changes item popularity and available evidence.

**What to do:**

1. Choose a movie with a low `ratings_count` in the popularity table.
2. Add one new rating for that movie.
3. Rebuild the matrix and rerun the popularity summary.
4. Compare the movie's count and mean rating before and after the change.
"""),
        challenge_code(2),
        md("""
### Your observations

Which movie changed most? Did one extra rating make the item easier to use in collaborative filtering? Why?

> Write your observations here:
>
> ................................................................................
>
> ................................................................................
>
> ................................................................................
"""),
        md("""
### Challenge 3 — Concept Check: Missing Ratings

This challenge requires **no programming**.

A user has not rated `Titanic`. The rating matrix shows this as a missing value.

Explain in your own words why this missing value should not be treated as a real rating of zero.

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


def notebook_02() -> list[dict]:
    return [
        md("""
# Chapter 3 Practical 02: Cosine, Pearson, and Jaccard Similarity

Learning objectives:
- Compute user-user similarity on co-rated items.
- Compare cosine and Pearson similarity for explicit ratings.
- Use Jaccard similarity for binary interactions.
- Use overlap counts to judge whether a similarity value is trustworthy.

Slide connection: similarity measures, cosine equation, Pearson equation, Jaccard similarity, and sparse overlap.
"""),
        code(COMMON_LOAD),
        md(PACKAGE_NOTE),
        md("""
## Technique: similarity on shared evidence

The helper functions below compare users only on items that both users rated. Missing values stay missing; they are not treated as zero ratings.

- `common_ratings()` uses `dropna(axis=1)` to keep only co-rated movies.
- `cosine_on_overlap()` uses `np.linalg.norm()` and `np.dot()` to compare rating vectors by angle.
- `pearson_on_overlap()` uses `np.corrcoef()` after checking that at least two co-rated items exist.
- `jaccard_liked()` first converts ratings into liked-item sets, then compares set overlap with set union.
"""),
        code(r"""
def common_ratings(matrix, user_a, user_b):
    pair = matrix.loc[[user_a, user_b]].dropna(axis=1)
    return pair.loc[user_a], pair.loc[user_b]

def cosine_on_overlap(matrix, user_a, user_b):
    a, b = common_ratings(matrix, user_a, user_b)
    if len(a) == 0:
        return np.nan
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return np.nan if denom == 0 else float(np.dot(a, b) / denom)

def pearson_on_overlap(matrix, user_a, user_b):
    a, b = common_ratings(matrix, user_a, user_b)
    if len(a) < 2:
        return np.nan
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def jaccard_liked(matrix, user_a, user_b, threshold=5):
    liked_a = set(matrix.columns[matrix.loc[user_a] >= threshold])
    liked_b = set(matrix.columns[matrix.loc[user_b] >= threshold])
    if not liked_a and not liked_b:
        return np.nan
    return len(liked_a & liked_b) / len(liked_a | liked_b)

print("Helper functions ready: common_ratings, cosine_on_overlap, pearson_on_overlap, jaccard_liked")
"""),
        md("""
The next cell applies the helper functions to one `target` user. The table shows how each similarity measure can tell a slightly different story about the same pair of users.
"""),
        code(r"""
rows = []
target = "Karen"
for other in rating_matrix.index.drop(target):
    overlap_count = rating_matrix.loc[[target, other]].notna().all(axis=0).sum()
    rows.append({
        "target_user": target,
        "other_user": other,
        "co_rated_items": int(overlap_count),
        "cosine": cosine_on_overlap(rating_matrix, target, other),
        "pearson": pearson_on_overlap(rating_matrix, target, other),
        "jaccard_liked": jaccard_liked(rating_matrix, target, other),
    })

similarities = pd.DataFrame(rows).sort_values("pearson", ascending=False)
similarities.round(3)
"""),
        md("""
## Technique: Pearson correlation matrix

Pearson removes each user's average rating behavior. This matters when one user rates generously and another rates strictly. The `similarity_matrix()` function applies one similarity function to every user pair and stores the result in a square `DataFrame`.
"""),
        code(r"""
def similarity_matrix(metric):
    users = rating_matrix.index
    sim = pd.DataFrame(index=users, columns=users, dtype=float)
    for u in users:
        for v in users:
            sim.loc[u, v] = metric(rating_matrix, u, v) if u != v else 1.0
    return sim

pearson_sim = similarity_matrix(pearson_on_overlap)
pearson_sim.round(2)
"""),
        md("Small overlap can make a similarity score unstable. Here we keep the raw similarity and show the overlap count next to it so students can judge the evidence."),
        code(r"""
similarities[["other_user", "co_rated_items", "cosine", "pearson", "jaccard_liked"]].round(3)
"""),
        md("""
# Challenges

### Challenge 1 — Change the Target User

**Goal:**
Investigate whether a different target user has different nearest neighbors.

**What to do:**

1. Change the existing `target` variable from `"Karen"` to another user, such as `"Alice"` or `"Sally"`.
2. Rerun the similarity table.
3. Compare cosine, Pearson, Jaccard, and `co_rated_items`.
4. Identify which neighbors changed.
"""),
        challenge_code(1),
        md("""
### Your observations

Which neighbors changed when you changed the target user? Did all similarity measures rank users in the same order? Which measure looked most reasonable?

> Write your observations here:
>
> ................................................................................
>
> ................................................................................
>
> ................................................................................
"""),
        md("""
### Challenge 2 — Change the Jaccard Like Threshold

**Goal:**
Investigate how the definition of "liked" changes binary Jaccard similarity.

**What to do:**

1. In `jaccard_liked`, change `threshold=5` to `threshold=6`.
2. Rerun the similarity table for the same target user.
3. Compare the Jaccard values before and after the change.
4. Explain why some values changed more than others.
"""),
        challenge_code(2),
        md("""
### Your observations

Which users had different Jaccard similarity values? Did cosine or Pearson change when the Jaccard threshold changed? Why?

> Write your observations here:
>
> ................................................................................
>
> ................................................................................
>
> ................................................................................
"""),
        md("""
### Challenge 3 — Concept Check: Small Overlap

This challenge requires **no programming**.

Two users have a very high similarity score, but they have only one or two co-rated movies.

Explain why this similarity value may be less reliable than a slightly lower score based on many co-rated movies.

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


def notebook_03() -> list[dict]:
    return [
        md("""
# Chapter 3 Practical 03: User-User kNN Prediction

Learning objectives:
- Select nearest neighbors for a target user.
- Predict missing ratings with a mean-centered, similarity-weighted formula.
- Explain a prediction through neighbor contributions.
- Generate Top-N user-user CF recommendations.

Slide connection: user-user CF, making predictions, prediction formula, and explainable memory-based CF.
"""),
        code(COMMON_LOAD),
        md(PACKAGE_NOTE),
        md("""
## Technique: finding user-user neighbors

User-user CF first finds users with similar rating patterns. In this notebook:

- `pearson_on_overlap()` compares two users only on co-rated movies.
- `user_neighbors()` loops over all other users, calculates overlap and similarity, then sorts the candidates.
- `min_overlap=2` avoids trusting a similarity score based on only one shared movie.

The output table is the candidate-neighbor list for the target user.
"""),
        code(r"""
def pearson_on_overlap(matrix, user_a, user_b):
    pair = matrix.loc[[user_a, user_b]].dropna(axis=1)
    if pair.shape[1] < 2:
        return np.nan
    if pair.loc[user_a].std() == 0 or pair.loc[user_b].std() == 0:
        return np.nan
    return float(np.corrcoef(pair.loc[user_a], pair.loc[user_b])[0, 1])

def user_neighbors(matrix, target_user, min_overlap=2):
    rows = []
    for other in matrix.index.drop(target_user):
        overlap = matrix.loc[[target_user, other]].notna().all(axis=0).sum()
        sim = pearson_on_overlap(matrix, target_user, other)
        if overlap >= min_overlap and not pd.isna(sim):
            rows.append({"neighbor": other, "similarity": sim, "overlap": int(overlap)})
    return pd.DataFrame(rows).sort_values("similarity", ascending=False)

user_neighbors(rating_matrix, "Karen")
"""),
        md("""
## Technique: mean-centered kNN rating prediction

The prediction starts from the target user's average rating. Each neighbor then contributes how much their rating for the target item is above or below their own average.

Important function details:

- `matrix.loc[target_user].mean()` calculates the target user's baseline rating.
- `neighbors.head(k_neighbors)` keeps only the top-k nearest neighbors.
- `abs(sim)` in the denominator follows the lecture formula and prevents positive and negative weights from canceling each other incorrectly.
- The returned `evidence` table explains the prediction by showing each neighbor's rating, mean rating, centered rating, and weighted contribution.
"""),
        code(r"""
def predict_user_user(matrix, target_user, target_item, k_neighbors=3, min_overlap=2, positive_only=True):
    target_mean = matrix.loc[target_user].mean()
    neighbors = user_neighbors(matrix, target_user, min_overlap=min_overlap)
    neighbors = neighbors[neighbors["neighbor"].map(lambda u: not pd.isna(matrix.loc[u, target_item]))]
    if positive_only:
        # In this introductory notebook, negative similarities are excluded because they indicate opposite taste.
        neighbors = neighbors[neighbors["similarity"] > 0]
    neighbors = neighbors.head(k_neighbors)
    if neighbors.empty:
        return np.nan, neighbors

    numerator = 0.0
    denominator = 0.0
    rows = []
    for _, row in neighbors.iterrows():
        u = row["neighbor"]
        sim = row["similarity"]
        neighbor_mean = matrix.loc[u].mean()
        centered_rating = matrix.loc[u, target_item] - neighbor_mean
        contribution = sim * centered_rating
        numerator += contribution
        denominator += abs(sim)
        rows.append({
            "neighbor": u,
            "similarity": sim,
            "neighbor_rating": matrix.loc[u, target_item],
            "neighbor_mean": neighbor_mean,
            "centered_rating": centered_rating,
            "weighted_contribution": contribution,
        })
    prediction = target_mean + numerator / denominator if denominator else np.nan
    return prediction, pd.DataFrame(rows)

target_user = "Karen"
target_item = "Independence Day"
k_neighbors = 3

pred, evidence = predict_user_user(rating_matrix, target_user, target_item, k_neighbors=k_neighbors)
print(f"Predicted Karen rating for Independence Day: {pred:.2f}")
evidence.round(3)
"""),
        md("""
## Technique: building a recommendation list

`recommend_user_user()` applies the same prediction function to every movie the target user has not rated. The final table is sorted by `predicted_rating`, so the highest-scoring unseen movies appear first.
"""),
        code(r"""
def recommend_user_user(matrix, target_user, n=5, k_neighbors=3):
    unseen_items = matrix.columns[matrix.loc[target_user].isna()]
    rows = []
    for item in unseen_items:
        pred, evidence = predict_user_user(matrix, target_user, item, k_neighbors=k_neighbors)
        if not pd.isna(pred):
            rows.append({
                "user": target_user,
                "recommended_movie": item,
                "predicted_rating": pred,
                "supporting_neighbors": ", ".join(evidence["neighbor"].tolist()),
            })
    return pd.DataFrame(rows).sort_values("predicted_rating", ascending=False).head(n)

recommend_user_user(rating_matrix, target_user, n=5, k_neighbors=k_neighbors).round(2)
"""),
        md("""
# Challenges

### Challenge 1 — Change the Neighborhood Size

**Goal:**
Investigate how the number of nearest neighbors changes a user-user prediction.

**What to do:**

1. Change the existing `k_neighbors` value from `3` to `1`.
2. Rerun the prediction and recommendation cells.
3. Change `k_neighbors` again to `5`.
4. Compare the predicted rating and supporting neighbors.
"""),
        challenge_code(1),
        md("""
### Your observations

How did the prediction change when `k_neighbors` changed? Which value gave the clearest explanation? Why can too few or too many neighbors be risky?

> Write your observations here:
>
> ................................................................................
>
> ................................................................................
>
> ................................................................................
"""),
        md("""
### Challenge 2 — Change the Target User and Item

**Goal:**
Investigate whether the same user-user CF method behaves differently for another prediction target.

**What to do:**

1. Change `target_user` from `"Karen"` to `"Alice"`.
2. Change `target_item` to an unseen movie for Alice, such as `"Toy Story"`.
3. Rerun the neighbor, prediction, and recommendation cells.
4. Compare the new evidence table with the original one.
"""),
        challenge_code(2),
        md("""
### Your observations

Did the selected neighbors change? Was there enough neighbor evidence for the new target item? How did the explanation differ from Karen's prediction?

> Write your observations here:
>
> ................................................................................
>
> ................................................................................
>
> ................................................................................
"""),
        md("""
### Challenge 3 — Concept Check: Negative Similarity

This challenge requires **no programming**.

In the notebook, negative similarities are excluded from the beginner prediction.

Explain what a negative user-user similarity means and why including it without explanation could confuse a simple rating prediction.

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


def notebook_04() -> list[dict]:
    return [
        md("""
# Chapter 3 Practical 04: Item-Item Collaborative Filtering

Learning objectives:
- Compute item-item similarities from user ratings.
- Predict a missing rating from similar items the user already rated.
- Compare item-item CF with user-user CF.
- Discuss why item-item CF is often easier to cache and serve.

Slide connection: item-item CF concept, item-item prediction example, and scalability.
"""),
        code(COMMON_LOAD),
        md(PACKAGE_NOTE),
        md("""
## Technique: item-item similarity matrix

Item-item CF compares columns of the user-item matrix. Two movies are compared only using users who rated both movies.

Important function details:

- `matrix[[item_a, item_b]].dropna()` keeps only users who rated both items.
- `np.corrcoef()` calculates Pearson correlation between the two item columns.
- The nested loop fills a square item-item similarity matrix that can be reused for many users.
"""),
        code(r"""
def item_pearson(matrix, item_a, item_b):
    pair = matrix[[item_a, item_b]].dropna()
    if len(pair) < 2:
        return np.nan
    if pair[item_a].std() == 0 or pair[item_b].std() == 0:
        return np.nan
    return float(np.corrcoef(pair[item_a], pair[item_b])[0, 1])

items = rating_matrix.columns
item_sim = pd.DataFrame(index=items, columns=items, dtype=float)
for a in items:
    for b in items:
        item_sim.loc[a, b] = 1.0 if a == b else item_pearson(rating_matrix, a, b)

item_sim.round(2)
"""),
        md("For one `target_item`, the next cell sorts its similarity column so students can inspect which movies are most related based on shared user ratings."),
        code(r"""
target_item = "Independence Day"
item_sim[target_item].drop(target_item).sort_values(ascending=False).round(3)
"""),
        md("""
## Technique: item-based rating prediction

To predict a user's missing rating, use similar items the same user has already rated.

- `matrix.loc[target_user].dropna()` finds the target user's known ratings.
- The evidence table keeps rated items, the user's rating for each item, and the item's similarity to the target item.
- `positive_only=True` skips negative similarities in the beginner version because they indicate opposite rating patterns.
- The prediction is a similarity-weighted average of ratings on related items.
"""),
        code(r"""
def predict_item_item(matrix, target_user, target_item, k_neighbors=3, positive_only=True):
    rated = matrix.loc[target_user].dropna()
    candidates = []
    for item, rating in rated.items():
        sim = item_sim.loc[target_item, item]
        if pd.isna(sim):
            continue
        if positive_only and sim <= 0:
            continue
        candidates.append({"rated_item": item, "rating": rating, "similarity": sim})
    evidence = pd.DataFrame(candidates, columns=["rated_item", "rating", "similarity"])
    evidence = evidence.sort_values("similarity", ascending=False).head(k_neighbors)
    if evidence.empty:
        return np.nan, evidence
    numerator = (evidence["rating"] * evidence["similarity"]).sum()
    denominator = evidence["similarity"].abs().sum()
    return numerator / denominator if denominator else np.nan, evidence

target_user = "Karen"
k_neighbors = 3

pred, evidence = predict_item_item(rating_matrix, target_user, target_item, k_neighbors=k_neighbors)
print(f"Predicted Karen rating for Independence Day: {pred:.2f}")
evidence.round(3)
"""),
        md("The recommender applies the same item-item prediction to every unseen movie and returns the highest predicted ratings. This is the item-item version of a top-N recommendation list."),
        code(r"""
def recommend_item_item(matrix, target_user, n=5, k_neighbors=3):
    unseen_items = matrix.columns[matrix.loc[target_user].isna()]
    rows = []
    for item in unseen_items:
        pred, evidence = predict_item_item(matrix, target_user, item, k_neighbors=k_neighbors, positive_only=True)
        if not pd.isna(pred):
            rows.append({
                "user": target_user,
                "recommended_movie": item,
                "predicted_rating": pred,
                "similar_rated_items": ", ".join(evidence["rated_item"].tolist()),
            })
    return pd.DataFrame(rows).sort_values("predicted_rating", ascending=False).head(n)

recommend_item_item(rating_matrix, "Karen").round(2)
"""),
        md("""
# Challenges

### Challenge 1 — Change the Target Item

**Goal:**
Investigate how item-item evidence changes when the target item changes.

**What to do:**

1. Change `target_item` from `"Independence Day"` to another unseen item for Karen.
2. Rerun the item similarity and prediction cells.
3. Inspect the `evidence` table.
4. Compare the similar rated items with the original result.
"""),
        challenge_code(1),
        md("""
### Your observations

Which rated items supported the new prediction? Were the similarities based on enough shared users? Did the predicted rating increase or decrease?

> Write your observations here:
>
> ................................................................................
>
> ................................................................................
>
> ................................................................................
"""),
        md("""
### Challenge 2 — Include Negative Item Similarities

**Goal:**
Investigate how negative item similarity can affect item-item predictions.

**What to do:**

1. In `predict_item_item`, call the function with `positive_only=False`.
2. Rerun the prediction for the same `target_user` and `target_item`.
3. Compare the evidence table with the positive-only version.
4. Explain whether the negative similarities made the result easier or harder to interpret.
"""),
        challenge_code(2),
        md("""
### Your observations

Did any negative similarities appear? How did the predicted rating change? Would you keep or exclude negative similarities for a beginner recommender?

> Write your observations here:
>
> ................................................................................
>
> ................................................................................
>
> ................................................................................
"""),
        md("""
### Challenge 3 — Concept Check: Item-Item Scalability

This challenge requires **no programming**.

Many platforms have millions of users but a smaller and more stable catalog of items.

Explain why item-item collaborative filtering can be easier to cache and serve than comparing a target user with every other user.

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


def notebook_05() -> list[dict]:
    return [
        md("""
# Chapter 3 Practical 05: Evaluation with MAE, Precision@K, and Recall@K

Learning objectives:
- Create a leave-one-out test split.
- Evaluate rating prediction with MAE.
- Evaluate Top-K recommendation with Precision@K, Recall@K, and HitRate@K.
- Compare user-user and item-item CF against a popularity baseline.

Slide connection: practical exercise evaluation requirements and performance comparison.
"""),
        code(COMMON_LOAD),
        md(PACKAGE_NOTE),
        md("""
## Technique: leave-one-out evaluation split

This notebook removes one rating per user as a small test set. The remaining ratings form `train_matrix`, which is used for prediction and recommendation. This keeps the evaluation simple: each user has one hidden item that the recommender tries to recover or predict.
"""),
        code(r"""
np.random.seed(7)
test_rows = ratings_named.groupby("user_id", group_keys=False).sample(n=1, random_state=7)
train_rows = ratings_named.drop(test_rows.index)
train_matrix = train_rows.pivot_table(index="user_id", columns="title", values="rating")

test_rows[["user_id", "title", "rating"]].sort_values("user_id")
"""),
        md("""
## Technique: prediction functions for evaluation

The next cell defines reusable prediction helpers:

- `predict_user_user_eval()` predicts a hidden rating from similar users.
- `build_item_sim()` creates item-item similarities from the training matrix.
- `predict_item_item_eval()` predicts a hidden rating from similar items the user already rated.

These functions are intentionally compact so students can focus on how evaluation calls them.
"""),
        code(r"""
def pearson_on_overlap(matrix, user_a, user_b):
    if user_a not in matrix.index or user_b not in matrix.index:
        return np.nan
    pair = matrix.loc[[user_a, user_b]].dropna(axis=1)
    if pair.shape[1] < 2:
        return np.nan
    if pair.loc[user_a].std() == 0 or pair.loc[user_b].std() == 0:
        return np.nan
    return float(np.corrcoef(pair.loc[user_a], pair.loc[user_b])[0, 1])

def predict_user_user_eval(matrix, target_user, item, k=3):
    if target_user not in matrix.index or item not in matrix.columns:
        return np.nan
    rows = []
    for other in matrix.index.drop(target_user):
        if pd.isna(matrix.loc[other, item]):
            continue
        sim = pearson_on_overlap(matrix, target_user, other)
        if not pd.isna(sim) and sim > 0:
            rows.append((other, sim))
    rows = sorted(rows, key=lambda x: x[1], reverse=True)[:k]
    if not rows:
        return np.nan
    target_mean = matrix.loc[target_user].mean()
    numerator = sum(sim * (matrix.loc[u, item] - matrix.loc[u].mean()) for u, sim in rows)
    denominator = sum(abs(sim) for _, sim in rows)
    return target_mean + numerator / denominator if denominator else np.nan

def build_item_sim(matrix):
    sim = pd.DataFrame(index=matrix.columns, columns=matrix.columns, dtype=float)
    for a in matrix.columns:
        for b in matrix.columns:
            pair = matrix[[a, b]].dropna()
            sim.loc[a, b] = 1.0 if a == b else (
                np.corrcoef(pair[a], pair[b])[0, 1]
                if len(pair) >= 2 and pair[a].std() != 0 and pair[b].std() != 0
                else np.nan
            )
    return sim

train_item_sim = build_item_sim(train_matrix)

def predict_item_item_eval(matrix, user, item, k=3):
    if user not in matrix.index or item not in train_item_sim.index:
        return np.nan
    rated = matrix.loc[user].dropna()
    rows = []
    for rated_item, rating in rated.items():
        sim = train_item_sim.loc[item, rated_item]
        if not pd.isna(sim) and sim > 0:
            rows.append((rated_item, rating, sim))
    rows = sorted(rows, key=lambda x: x[2], reverse=True)[:k]
    if not rows:
        return np.nan
    numerator = sum(rating * sim for _, rating, sim in rows)
    denominator = sum(abs(sim) for _, _, sim in rows)
    return numerator / denominator if denominator else np.nan

print("Evaluation helper functions ready: predict_user_user_eval, build_item_sim, predict_item_item_eval")
"""),
        md("""
## Technique: MAE for rating prediction

Mean Absolute Error (MAE) compares predicted ratings with the hidden actual ratings. Lower MAE means the predicted ratings are closer to the true held-out ratings.
"""),
        code(r"""
def popularity_prediction(matrix, item):
    return matrix[item].mean() if item in matrix.columns else matrix.stack().mean()

pred_rows = []
for _, row in test_rows.iterrows():
    user, item, actual = row["user_id"], row["title"], row["rating"]
    for model, pred in [
        ("user_user", predict_user_user_eval(train_matrix, user, item)),
        ("item_item", predict_item_item_eval(train_matrix, user, item)),
        ("popularity", popularity_prediction(train_matrix, item)),
    ]:
        if not pd.isna(pred):
            pred_rows.append({"model": model, "user": user, "item": item, "actual": actual, "predicted": pred})

predictions = pd.DataFrame(pred_rows)
predictions["absolute_error"] = (predictions["actual"] - predictions["predicted"]).abs()
predictions.groupby("model")["absolute_error"].mean().rename("MAE").round(3)
"""),
        md("""
## Technique: Precision@K, Recall@K, and HitRate@K

Top-k metrics evaluate the ranked recommendation list rather than a single rating prediction.

- `topk_popularity()` recommends unseen movies with the highest average rating.
- `topk_user_user()` recommends unseen movies with the highest predicted user-user score.
- `evaluate_topk()` compares the recommended list with held-out relevant items.
- Precision@K asks: out of K recommended items, how many were relevant?
- Recall@K asks: out of all relevant hidden items, how many did we recover?
- HitRate@K asks: did we get at least one relevant item?
"""),
        code(r"""
def topk_popularity(matrix, user, k=3):
    seen = set(matrix.loc[user].dropna().index)
    scores = matrix.mean().drop(labels=list(seen), errors="ignore")
    return scores.sort_values(ascending=False).head(k).index.tolist()

def topk_user_user(matrix, user, k=3):
    unseen = matrix.columns[matrix.loc[user].isna()]
    rows = [(item, predict_user_user_eval(matrix, user, item)) for item in unseen]
    rows = [(item, score) for item, score in rows if not pd.isna(score)]
    return [item for item, _ in sorted(rows, key=lambda x: x[1], reverse=True)[:k]]

def evaluate_topk(model_fn, train_matrix, test_rows, k=3, relevant_threshold=5):
    rows = []
    for user, user_test in test_rows.groupby("user_id"):
        relevant = set(user_test.loc[user_test["rating"] >= relevant_threshold, "title"])
        recommended = model_fn(train_matrix, user, k)
        hits = len(set(recommended) & relevant)
        rows.append({
            "user": user,
            "precision_at_k": hits / k,
            "recall_at_k": hits / len(relevant) if relevant else np.nan,
            "hit_rate_at_k": 1 if hits > 0 else 0,
            "recommended": recommended,
            "relevant": sorted(relevant),
        })
    return pd.DataFrame(rows)

pop_eval = evaluate_topk(topk_popularity, train_matrix, test_rows)
uu_eval = evaluate_topk(topk_user_user, train_matrix, test_rows)

pd.DataFrame({
    "model": ["popularity", "user_user"],
    "precision_at_3": [pop_eval["precision_at_k"].mean(), uu_eval["precision_at_k"].mean()],
    "recall_at_3": [pop_eval["recall_at_k"].mean(), uu_eval["recall_at_k"].mean()],
    "hit_rate_at_3": [pop_eval["hit_rate_at_k"].mean(), uu_eval["hit_rate_at_k"].mean()],
}).round(3)
"""),
        md("""
# Challenges

### Challenge 1 — Change the Relevance Threshold

**Goal:**
Investigate how the definition of a relevant item changes Precision@K and Recall@K.

**What to do:**

1. In `evaluate_topk`, change `relevant_threshold` from `5` to `6`.
2. Rerun the popularity and user-user evaluation cells.
3. Compare Precision@3, Recall@3, and HitRate@3 with the original result.
4. Explain why stricter relevance can change the metrics.
"""),
        challenge_code(1),
        md("""
### Your observations

Which metric changed most? Did both models change in the same way? What does a stricter relevance threshold mean for evaluation?

> Write your observations here:
>
> ................................................................................
>
> ................................................................................
>
> ................................................................................
"""),
        md("""
### Challenge 2 — Change the Recommendation List Length

**Goal:**
Investigate how evaluating a longer recommendation list changes top-k metrics.

**What to do:**

1. Change the evaluation value of `k` from `3` to `5`.
2. Rerun `evaluate_topk` for the popularity and user-user recommenders.
3. Update the summary table labels so they describe `Precision@5`, `Recall@5`, and `HitRate@5`.
4. Compare the new metrics with the original @3 result.
"""),
        challenge_code(2),
        md("""
### Your observations

Did Precision decrease or increase when `k` became larger? Did Recall change differently from Precision? Why can this happen?

> Write your observations here:
>
> ................................................................................
>
> ................................................................................
>
> ................................................................................
"""),
        md("""
### Challenge 3 — Concept Check: Rating Error vs Top-K Quality

This challenge requires **no programming**.

A model has a low MAE, but its top-3 recommendation list contains no relevant items for a user.

Explain why rating-prediction accuracy and recommendation-list quality are related but not the same thing.

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


def notebook_06() -> list[dict]:
    return [
        md("""
# Chapter 3 Practical 06: Cold Start, Sparsity, Clustering, and Temporal Dynamics

Learning objectives:
- Detect cold-start users and items.
- Discuss sparsity, popularity bias, and cold start.
- Cluster users to reduce neighbor search.
- Add time-decay weights to recent interactions.
- Complete three short Chapter 3 challenges.

Slide connection: limits of memory-based CF, practical mitigations, clustering for scalability, and temporal dynamics.
"""),
        code(COMMON_LOAD),
        md(PACKAGE_NOTE),
        md("Cold-start users and cold-start items have too few interactions for reliable memory-based CF."),
        code(r"""
user_counts = ratings_named.groupby("user_id").size().rename("ratings_count")
item_counts = ratings_named.groupby("title").size().rename("ratings_count")

print("Cold-start-like users:")
display(user_counts[user_counts <= 2])
print("Cold-start-like items:")
display(item_counts[item_counts <= 2])
"""),
        md("""
## Technique: sparsity and popularity bias

Sparsity means that most user-item pairs are unknown. Popularity bias means popular items get more evidence and may be recommended more often.

Important functions:

- `groupby("title")["rating"].agg(...)` calculates item-level counts and means.
- `notna().sum().sum()` counts observed ratings.
- Sorting by `ratings_count` helps identify items that dominate the available evidence.
"""),
        code(r"""
n_users, n_items = rating_matrix.shape
known_ratings = rating_matrix.notna().sum().sum()
sparsity = 1 - known_ratings / (n_users * n_items)

popularity = (
    ratings_named.groupby("title")["rating"]
    .agg(ratings_count="count", mean_rating="mean")
    .sort_values(["ratings_count", "mean_rating"], ascending=False)
)

print(f"Matrix sparsity: {sparsity:.1%}")
popularity.head(5).round(2)
"""),
        md("""
## Technique: clustering to reduce neighbor search

One scalability mitigation is to compare the target user only with users in the same cluster.

This section uses `scikit-learn`:

- `StandardScaler` puts filled rating columns on a comparable scale.
- `KMeans` groups users with similar filled rating profiles.
- Missing values are filled with item means only for clustering. They are not treated as real ratings for CF prediction.
"""),
        code(r"""
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

filled = rating_matrix.apply(lambda col: col.fillna(col.mean()), axis=0)
scaled = StandardScaler().fit_transform(filled)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = pd.Series(kmeans.fit_predict(scaled), index=rating_matrix.index, name="cluster")
clusters.to_frame().sort_values("cluster")
"""),
        code(r"""
target_user = "Karen"
target_cluster = clusters.loc[target_user]
candidate_neighbors = clusters[clusters.eq(target_cluster)].index.drop(target_user)
print(f"Compare Karen only with users in cluster {target_cluster}: {candidate_neighbors.tolist()}")
"""),
        md("""
## Technique: temporal dynamics and time decay

Temporal dynamics give more weight to recent interactions. A larger decay value makes older ratings fade more quickly.

Important functions:

- `np.exp(-decay_lambda * days_ago)` creates a smooth time-decay weight.
- `np.average(..., weights=...)` calculates a weighted mean rating.
- The `recent_profile` table shows which movies look stronger when recent ratings count more.
"""),
        code(r"""
decay_lambda = 0.01
ratings_named["time_weight"] = np.exp(-decay_lambda * ratings_named["days_ago"])
ratings_named["weighted_rating"] = ratings_named["rating"] * ratings_named["time_weight"]

ratings_named[["user_id", "title", "rating", "days_ago", "time_weight", "weighted_rating"]].sort_values("days_ago").head(10).round(3)
"""),
        code(r"""
recent_profile = (
    ratings_named.groupby("title")
    .apply(lambda g: np.average(g["rating"], weights=g["time_weight"]))
    .rename("time_weighted_mean")
    .sort_values(ascending=False)
)
recent_profile.head(5).round(2)
"""),
        md("""
# Challenges

### Challenge 1 — Change the Number of Clusters

**Goal:**
Investigate how clustering changes the reduced neighbor search space.

**What to do:**

1. Change `n_clusters=3` in the `KMeans` cell to `2`.
2. Rerun the clustering and candidate-neighbor cells.
3. Change `n_clusters` to `4` and rerun the same cells.
4. Compare the candidate neighbors for `target_user`.
"""),
        challenge_code(1),
        md("""
### Your observations

How did the candidate-neighbor list change? Which cluster setting gave a useful search space? Why can too many or too few clusters be a problem?

> Write your observations here:
>
> ................................................................................
>
> ................................................................................
>
> ................................................................................
"""),
        md("""
### Challenge 2 — Change the Time-Decay Rate

**Goal:**
Investigate how stronger or weaker time decay changes the time-weighted popularity ranking.

**What to do:**

1. Change `decay_lambda` from `0.01` to `0.03`.
2. Rerun the time-weight and `recent_profile` cells.
3. Change `decay_lambda` to `0.001` and rerun the same cells.
4. Compare which movies move up or down in the time-weighted ranking.
"""),
        challenge_code(2),
        md("""
### Your observations

Which movies benefited most from stronger time decay? Which older interactions lost influence? When would time decay be useful in a recommender?

> Write your observations here:
>
> ................................................................................
>
> ................................................................................
>
> ................................................................................
"""),
        md("""
### Challenge 3 — Concept Check: Cold Start and Popularity Bias

This challenge requires **no programming**.

> A new user joins a movie platform but has not rated, liked, or watched any movies.
> At the same time, a newly released movie has not yet received any interactions.

Explain in your own words:

1. Why can standard memory-based collaborative filtering not provide reliable personalized recommendations in these two cases?
2. Suggest one practical solution for the new user.
3. Suggest one practical solution for the new item.
4. Why can always falling back to popular items reduce recommendation diversity?

### Your explanation
>
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
        """# Chapter 3: Collaborative Filtering - Memory-Based

This folder contains the reorganized practical material for Chapter 3. The notebooks move from the user-item matrix to user-user CF, item-item CF, evaluation, and practical challenges.

## Notebook Path

| Order | Notebook | Main idea |
| --- | --- | --- |
| 01 | `01_user_item_matrix_and_sparsity.ipynb` | User-item matrix, missing values, density, sparsity, co-rating overlap |
| 02 | `02_similarity_measures_for_cf.ipynb` | Cosine, Pearson, Jaccard, and co-rated-item overlap |
| 03 | `03_user_user_knn_prediction.ipynb` | User-user nearest neighbors, mean-centered weighted prediction, explanations |
| 04 | `04_item_item_collaborative_filtering.ipynb` | Item-item similarity, item-based prediction, cached similarity intuition |
| 05 | `05_evaluation_precision_recall_mae.ipynb` | Leave-one-out split, MAE, Precision@K, Recall@K, HitRate@K |
| 06 | `06_cold_start_sparsity_clustering_temporal.ipynb` | Cold start, sparsity, popularity bias, clustering, time decay, and the three Chapter 3 challenges |

These six notebooks are the single recommended student path for Chapter 3. Advanced graph-based material has been moved out of the student path because it is no longer part of the revised chapter.

## Data

The `data/` folder contains small local CSV files used across the notebooks:

- `movies_chapter3.csv`
- `ratings_chapter3.csv`

The examples use a compact movie-rating matrix based on the Chapter 3 lecture examples, with a few extra users and movies for evaluation, clustering, and temporal dynamics.

## HTML Demos

The `html_demos/` folder contains one standalone real-world classroom demo:

- `index.html`

This single page gives a compact overview of the revised Chapter 3 practical path and links students to the six notebooks. It can be opened directly in a browser.

## Optional Dependencies

The core notebooks use `pandas`, `numpy`, `scikit-learn`, and `matplotlib`. The examples are deliberately small so students can inspect intermediate tables and understand each step.
""",
        encoding="utf-8",
    )


def write_report() -> None:
    (CHAPTER / "CHAPTER_03_REORGANIZATION_REPORT.md").write_text(
        """# Chapter 3 Reorganization Report

## Source Alignment

The structure is based on `RS_C3_V4.pdf`, Chapter 3: Collaborative Filtering - Memory-Based. The revised deck topics include:

- Why collaborative filtering and when it works.
- Memory-based collaborative filtering.
- User-user and item-item CF.
- User-item rating matrices and sparsity.
- Cosine, Pearson, and Jaccard similarity.
- Mean-centered kNN prediction.
- Evaluation with Precision@K, Recall@K, and MAE.
- Cold start, sparsity, popularity bias, scalability, and practical mitigations.
- Clustering to reduce the neighbor search space.
- Temporal dynamics and time-decay weighting.

## Repository Findings

The repository already contained two root-level Chapter 3 notebooks:

- `Chapter3_CF_Practical.ipynb`
- `Chapter3_CF_Practical_Enhanced.ipynb`

Those notebooks were moved into `chapter_03_collaborative_filtering/archive/` as legacy reference material, because the chapter folder should expose one clear student path.

## Implemented Structure

Created:

```text
chapter_03_collaborative_filtering/
  01_user_item_matrix_and_sparsity.ipynb
  02_similarity_measures_for_cf.ipynb
  03_user_user_knn_prediction.ipynb
  04_item_item_collaborative_filtering.ipynb
  05_evaluation_precision_recall_mae.ipynb
  06_cold_start_sparsity_clustering_temporal.ipynb
  archive/
    Chapter3_CF_Practical_legacy.ipynb
    Chapter3_CF_Practical_Enhanced_legacy.ipynb
    07_graph_ppr_explainable_hybrid_cf.ipynb
  data/
  html_demos/
  README.md
  CHAPTER_03_REORGANIZATION_REPORT.md
```

## Design Choices

- Kept the practical path beginner-friendly and sequential, mirroring the Chapter 2 organization.
- Used local CSV files so notebooks run without remote downloads.
- Used a compact rating matrix close to the lecture examples, with enough additional interactions for evaluation, clustering, and temporal dynamics.
- Kept only memory-based CF topics from the revised slides: cosine, Pearson, Jaccard, top-k neighbors, user-user prediction, item-item prediction, evaluation, cold start, sparsity, popularity bias, clustering, and temporal weighting.
- Removed graph-based CF, Personalized PageRank, path explanations, and hybrid graph fallback from the student path.
- Added the three requested student challenges: neighborhood size, cosine versus Pearson, and cold-start concept check.
- Archived the existing root-level Chapter 3 notebooks while keeping the Chapter 3 folder focused on the six practical notebooks.

## Suggested Cleanup Later

- If the course uses MovieLens in class, optionally add one advanced notebook that repeats the same pipeline on `Datasets/ml-latest-small`.
- The archived graph/PPR notebook can be reused in a future advanced chapter if graph-based collaborative filtering returns to the course.
""",
        encoding="utf-8",
    )


HTML_STYLE = """
<style>
  :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  body { margin: 0; background: #f7f7f3; color: #222; }
  main { max-width: 1120px; margin: 0 auto; padding: 32px 20px 48px; }
  h1 { margin: 0 0 10px; font-size: clamp(2rem, 4vw, 3.4rem); letter-spacing: 0; }
  h2 { margin-top: 32px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; }
  .card { background: white; border: 1px solid #deded4; border-radius: 8px; padding: 16px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
  a { color: #0b6f6a; font-weight: 700; text-decoration: none; }
  table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; }
  th, td { border: 1px solid #ddd; padding: 10px; text-align: center; }
  th { background: #1f2933; color: white; }
  .missing { background: #f0eee7; color: #9a9387; }
  .high { background: #b7e4c7; }
  .mid { background: #fff3b0; }
  .low { background: #f4b4a6; }
  button, select { font: inherit; border: 1px solid #b7b7aa; background: white; border-radius: 6px; padding: 8px 10px; }
  .bar { height: 16px; background: #2a9d8f; border-radius: 4px; }
  .node { display: inline-flex; align-items: center; justify-content: center; min-width: 92px; min-height: 42px; margin: 6px; border-radius: 999px; border: 2px solid #64748b; background: white; }
  .user { border-color: #0f766e; }
  .item { border-color: #b45309; }
  .edge { color: #555; font-size: .9rem; }
</style>
"""


def write_html() -> None:
    DEMOS.mkdir(parents=True, exist_ok=True)
    for stale_demo in DEMOS.glob("*.html"):
        if stale_demo.name != "index.html":
            stale_demo.unlink()
    index = DEMOS / "index.html"
    index.write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Chapter 3 Memory-Based Collaborative Filtering</title>
  <style>
    :root { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2933; background: #f6f7f9; }
    body { margin: 0; }
    main { max-width: 980px; margin: 0 auto; padding: 32px 18px 44px; }
    h1 { margin: 0 0 10px; font-size: clamp(2rem, 5vw, 3.6rem); letter-spacing: 0; }
    p { line-height: 1.55; color: #52606d; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; margin-top: 22px; }
    a { display: block; min-height: 118px; padding: 16px; border: 1px solid #d9e2ec; border-radius: 8px; background: #fff; color: #102a43; text-decoration: none; }
    strong { display: block; margin-bottom: 8px; }
    span { color: #627d98; line-height: 1.45; }
  </style>
</head>
<body>
  <main>
    <h1>Chapter 3: Memory-Based Collaborative Filtering</h1>
    <p>Use these notebooks in order. The revised path focuses on user-item matrices, cosine, Pearson, Jaccard, user-user CF, item-item CF, simple evaluation, cold start, sparsity, clustering, and temporal dynamics.</p>
    <section class="grid">
      <a href="../01_user_item_matrix_and_sparsity.ipynb"><strong>01 Matrix and Sparsity</strong><span>Load ratings, build the user-item matrix, and inspect missing values.</span></a>
      <a href="../02_similarity_measures_for_cf.ipynb"><strong>02 Similarity Measures</strong><span>Compare cosine, Pearson, and Jaccard on appropriate shared evidence.</span></a>
      <a href="../03_user_user_knn_prediction.ipynb"><strong>03 User-User CF</strong><span>Find top-k neighbors and predict a missing rating.</span></a>
      <a href="../04_item_item_collaborative_filtering.ipynb"><strong>04 Item-Item CF</strong><span>Recommend items related to those already liked.</span></a>
      <a href="../05_evaluation_precision_recall_mae.ipynb"><strong>05 Evaluation</strong><span>Use MAE, Precision@K, Recall@K, and HitRate@K.</span></a>
      <a href="../06_cold_start_sparsity_clustering_temporal.ipynb"><strong>06 Practical Challenges</strong><span>Explore neighborhood size, similarity choice, cold start, clustering, and time decay.</span></a>
    </section>
  </main>
</body>
</html>""",
        encoding="utf-8",
    )


def build() -> None:
    write_data()
    write_notebook("01_user_item_matrix_and_sparsity.ipynb", notebook_01())
    write_notebook("02_similarity_measures_for_cf.ipynb", notebook_02())
    write_notebook("03_user_user_knn_prediction.ipynb", notebook_03())
    write_notebook("04_item_item_collaborative_filtering.ipynb", notebook_04())
    write_notebook("05_evaluation_precision_recall_mae.ipynb", notebook_05())
    write_notebook("06_cold_start_sparsity_clustering_temporal.ipynb", notebook_06())
    write_html()
    write_readme()
    write_report()


if __name__ == "__main__":
    build()
