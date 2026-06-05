from __future__ import annotations

import csv
import json
from itertools import count
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "chapter_02_content_based"
DATA = CHAPTER / "data"
DEMOS = CHAPTER / "html_demos"
EXERCISES = CHAPTER / "slide_exercises"


MOVIES = [
    {
        "movie_id": 1,
        "title": "Inception",
        "genres": "Sci-Fi|Thriller|Action",
        "director": "Christopher Nolan",
        "year": 2010,
        "duration_min": 148,
        "rating": 8.8,
        "family_friendly": 0,
        "description": "A thief enters layered dreams to plant an idea while questioning what is real.",
        "keywords": "dreams heist subconscious mind-bending",
    },
    {
        "movie_id": 2,
        "title": "Interstellar",
        "genres": "Sci-Fi|Adventure|Drama",
        "director": "Christopher Nolan",
        "year": 2014,
        "duration_min": 169,
        "rating": 8.7,
        "family_friendly": 0,
        "description": "Astronauts travel through a wormhole to find a new home for humanity.",
        "keywords": "space exploration wormhole survival family",
    },
    {
        "movie_id": 3,
        "title": "Titanic",
        "genres": "Romance|Drama",
        "director": "James Cameron",
        "year": 1997,
        "duration_min": 195,
        "rating": 7.9,
        "family_friendly": 0,
        "description": "A young couple from different social classes fall in love on a doomed ocean liner.",
        "keywords": "romance ship tragedy historical",
    },
    {
        "movie_id": 4,
        "title": "The Matrix",
        "genres": "Sci-Fi|Action",
        "director": "The Wachowskis",
        "year": 1999,
        "duration_min": 136,
        "rating": 8.7,
        "family_friendly": 0,
        "description": "A hacker discovers that reality is a simulated world controlled by machines.",
        "keywords": "simulation hacker reality action cyberpunk",
    },
    {
        "movie_id": 5,
        "title": "Toy Story",
        "genres": "Animation|Adventure|Comedy|Family",
        "director": "John Lasseter",
        "year": 1995,
        "duration_min": 81,
        "rating": 8.3,
        "family_friendly": 1,
        "description": "A cowboy doll feels threatened when a space ranger toy becomes the new favorite.",
        "keywords": "toys friendship family adventure",
    },
    {
        "movie_id": 6,
        "title": "Finding Nemo",
        "genres": "Animation|Adventure|Family",
        "director": "Andrew Stanton",
        "year": 2003,
        "duration_min": 100,
        "rating": 8.2,
        "family_friendly": 1,
        "description": "A cautious clownfish crosses the ocean to rescue his lost son.",
        "keywords": "ocean rescue family journey fish",
    },
    {
        "movie_id": 7,
        "title": "The Dark Knight",
        "genres": "Action|Crime|Drama",
        "director": "Christopher Nolan",
        "year": 2008,
        "duration_min": 152,
        "rating": 9.0,
        "family_friendly": 0,
        "description": "Batman faces a chaotic criminal who challenges his moral code.",
        "keywords": "superhero crime chaos justice",
    },
    {
        "movie_id": 8,
        "title": "The Martian",
        "genres": "Sci-Fi|Adventure|Comedy",
        "director": "Ridley Scott",
        "year": 2015,
        "duration_min": 144,
        "rating": 8.0,
        "family_friendly": 0,
        "description": "An astronaut stranded on Mars uses science and humor to survive until rescue.",
        "keywords": "mars survival science astronaut space",
    },
    {
        "movie_id": 9,
        "title": "The Notebook",
        "genres": "Romance|Drama",
        "director": "Nick Cassavetes",
        "year": 2004,
        "duration_min": 123,
        "rating": 7.8,
        "family_friendly": 0,
        "description": "An elderly man reads a story about a lifelong romance across social barriers.",
        "keywords": "romance memory love drama",
    },
    {
        "movie_id": 10,
        "title": "Paddington",
        "genres": "Comedy|Family|Adventure",
        "director": "Paul King",
        "year": 2014,
        "duration_min": 95,
        "rating": 7.3,
        "family_friendly": 1,
        "description": "A polite young bear finds a home with a London family and spreads kindness.",
        "keywords": "family comedy london kindness adventure",
    },
    {
        "movie_id": 11,
        "title": "Gravity",
        "genres": "Sci-Fi|Thriller|Drama",
        "director": "Alfonso Cuaron",
        "year": 2013,
        "duration_min": 91,
        "rating": 7.7,
        "family_friendly": 0,
        "description": "Two astronauts struggle to survive after debris destroys their shuttle.",
        "keywords": "space survival astronaut disaster orbit",
    },
    {
        "movie_id": 12,
        "title": "La La Land",
        "genres": "Romance|Drama|Music",
        "director": "Damien Chazelle",
        "year": 2016,
        "duration_min": 128,
        "rating": 8.0,
        "family_friendly": 0,
        "description": "A jazz musician and an aspiring actor fall in love while pursuing artistic dreams.",
        "keywords": "music romance ambition hollywood",
    },
]


POSTER_URLS = {
    "Inception": "https://image.tmdb.org/t/p/w500/oYuLEt3zVCKq57qu2F8dT7NIa6f.jpg",
    "Interstellar": "https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
    "Titanic": "https://image.tmdb.org/t/p/w500/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg",
    "The Matrix": "https://image.tmdb.org/t/p/w500/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg",
    "Toy Story": "https://image.tmdb.org/t/p/w500/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg",
    "Finding Nemo": "https://image.tmdb.org/t/p/w500/eHuGQ10FUzK1mdOY69wF5pGgEf5.jpg",
    "The Dark Knight": "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
    "The Martian": "https://image.tmdb.org/t/p/w500/5BHuvQ6p9kfc091Z8RiFNhCwL4b.jpg",
    "The Notebook": "https://image.tmdb.org/t/p/w500/qom1SZSENdmHFNZBXbtJAU0WTlC.jpg",
    "Paddington": "https://image.tmdb.org/t/p/w500/wpchRGhRhvhtU083PfX2yixXtiw.jpg",
    "Gravity": "https://image.tmdb.org/t/p/w500/kZ2nZw8D681aphje8NJi8EfbL1U.jpg",
    "La La Land": "https://image.tmdb.org/t/p/w500/uDO8zWDhfWwoFdKS4fzkUJt0Rf0.jpg",
}


CELL_COUNTER = count(1)


INTERACTIONS = [
    {"user_id": "U1", "title": "Inception", "rating": 5, "clicked": 1, "liked": 1, "watch_minutes": 140, "days_ago": 3},
    {"user_id": "U1", "title": "Interstellar", "rating": 5, "clicked": 1, "liked": 1, "watch_minutes": 165, "days_ago": 12},
    {"user_id": "U1", "title": "The Matrix", "rating": 4, "clicked": 1, "liked": 1, "watch_minutes": 120, "days_ago": 30},
    {"user_id": "U2", "title": "Toy Story", "rating": 5, "clicked": 1, "liked": 1, "watch_minutes": 80, "days_ago": 2},
    {"user_id": "U2", "title": "Finding Nemo", "rating": 4, "clicked": 1, "liked": 1, "watch_minutes": 88, "days_ago": 8},
    {"user_id": "U2", "title": "Paddington", "rating": 5, "clicked": 1, "liked": 1, "watch_minutes": 90, "days_ago": 20},
    {"user_id": "U3", "title": "Titanic", "rating": 5, "clicked": 1, "liked": 1, "watch_minutes": 180, "days_ago": 5},
    {"user_id": "U3", "title": "The Notebook", "rating": 4, "clicked": 1, "liked": 1, "watch_minutes": 115, "days_ago": 11},
    {"user_id": "U3", "title": "La La Land", "rating": 4, "clicked": 1, "liked": 1, "watch_minutes": 110, "days_ago": 28},
]


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


COMMON_LOAD = r"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
if not (DATA_DIR / "movies_chapter2.csv").exists():
    DATA_DIR = Path("../data")
if not (DATA_DIR / "movies_chapter2.csv").exists():
    DATA_DIR = Path("chapter_02_content_based/data")

movies = pd.read_csv(DATA_DIR / "movies_chapter2.csv")
movies.head()
"""


def write_notebook(name: str, cells: list[dict], folder: Path = CHAPTER) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(nb(cells), indent=2), encoding="utf-8")


def write_data() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    DEMOS.mkdir(parents=True, exist_ok=True)
    EXERCISES.mkdir(parents=True, exist_ok=True)
    with (DATA / "movies_chapter2.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(MOVIES[0]))
        writer.writeheader()
        writer.writerows(MOVIES)
    with (DATA / "user_interactions_chapter2.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(INTERACTIONS[0]))
        writer.writeheader()
        writer.writerows(INTERACTIONS)


def notebook_01() -> list[dict]:
    return [
        md("""
# Chapter 2 Practical 01: Feature Vectors and Similarity

Learning objectives:
- Represent movies with numeric, categorical, and multi-label features.
- Compute cosine similarity, Jaccard similarity, and Euclidean distance.
- Interpret what each similarity measure means.
- Plot items in a simple 2D feature space.

Slide connection: item representation, feature vectors, vector space, cosine similarity, Jaccard similarity, and Euclidean distance.
"""),
        md("We start with five familiar movies and a few hand-written content features. Small examples make the vector idea easier to see."),
        code(r"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

movies = pd.DataFrame({
    "title": ["Inception", "Interstellar", "Titanic", "The Matrix", "Toy Story"],
    "duration_min": [148, 169, 195, 136, 81],
    "rating": [8.8, 8.7, 7.9, 8.7, 8.3],
    "genres": [
        ["Sci-Fi", "Thriller", "Action"],
        ["Sci-Fi", "Adventure", "Drama"],
        ["Romance", "Drama"],
        ["Sci-Fi", "Action"],
        ["Animation", "Adventure", "Comedy", "Family"],
    ],
})
movies
"""),
        md("Numeric features need scaling because duration and rating live on different ranges."),
        code(r"""
scaler = MinMaxScaler()
numeric_features = pd.DataFrame(
    scaler.fit_transform(movies[["duration_min", "rating"]]),
    columns=["duration_scaled", "rating_scaled"],
    index=movies["title"],
)
numeric_features.round(2)
"""),
        md("Genres are multi-label categorical features: one movie can belong to several genres. We encode each genre as a 0/1 column."),
        code(r"""
mlb = MultiLabelBinarizer()
genre_features = pd.DataFrame(
    mlb.fit_transform(movies["genres"]),
    columns=mlb.classes_,
    index=movies["title"],
)
genre_features
"""),
        md("Now we combine numeric and genre features into one item-feature matrix."),
        code(r"""
feature_matrix = pd.concat([numeric_features, genre_features], axis=1)
feature_matrix.round(2)
"""),
        md("Cosine similarity compares the angle between vectors. It is often useful when the pattern of features matters more than the raw size of the vector."),
        code(r"""
cosine = pd.DataFrame(
    cosine_similarity(feature_matrix),
    index=feature_matrix.index,
    columns=feature_matrix.index,
)
cosine.round(2)
"""),
        md("Jaccard similarity compares overlap between sets. Here we use it only for genre sets."),
        code(r"""
def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)

jaccard_rows = []
for other in movies["title"]:
    base_genres = movies.loc[movies["title"] == "Inception", "genres"].iloc[0]
    other_genres = movies.loc[movies["title"] == other, "genres"].iloc[0]
    jaccard_rows.append({"movie": other, "jaccard_with_inception": jaccard(base_genres, other_genres)})

pd.DataFrame(jaccard_rows).sort_values("jaccard_with_inception", ascending=False)
"""),
        md("Euclidean distance measures straight-line distance. Smaller means more similar, so we sort ascending."),
        code(r"""
distances = pd.DataFrame(
    euclidean_distances(feature_matrix),
    index=feature_matrix.index,
    columns=feature_matrix.index,
)
distances["Inception"].rename("distance_from_inception").sort_values().round(2)
"""),
        md("A 2D plot cannot show all features, but it helps students see the intuition of distance in a feature space."),
        code(r"""
plot_df = feature_matrix[["Sci-Fi", "Romance"]].copy()
plot_df["title"] = plot_df.index

ax = plot_df.plot.scatter(x="Sci-Fi", y="Romance", s=120, figsize=(6, 4))
for _, row in plot_df.iterrows():
    ax.text(row["Sci-Fi"] + 0.02, row["Romance"] + 0.02, row["title"])
ax.set_title("Movies in a tiny 2D genre space")
ax.set_xlim(-0.1, 1.25)
ax.set_ylim(-0.1, 1.25)
plt.show()
"""),
        md("""
## What did we learn?

- Feature vectors turn item metadata into numbers.
- One-hot and multi-hot encoding make categorical features usable.
- Cosine, Jaccard, and Euclidean measures answer related but different similarity questions.

Exercises:
1. Add one more movie and recompute all three similarities.
2. Change the numeric scaling or remove numeric features. Which recommendations change?
"""),
    ]


def notebook_02() -> list[dict]:
    return [
        md("""
# Chapter 2 Practical 02: TF-IDF Movie Recommender

Learning objectives:
- Clean and combine text metadata.
- Build Bag-of-Words and TF-IDF item vectors.
- Compute a cosine similarity matrix.
- Explain recommendations with shared terms.

Slide connection: Bag-of-Words, TF-IDF, vector normalization, cosine similarity, ranking, and Top-N recommendation.
"""),
        md("Load the small Chapter 2 movie dataset. It includes titles, genres, descriptions, and keywords."),
        code(COMMON_LOAD),
        md("We combine several text fields. This gives the recommender more content evidence than title or genre alone."),
        code(r"""
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

movies["combined_text"] = (
    movies["title"] + " " +
    movies["genres"].str.replace("|", " ", regex=False) + " " +
    movies["director"] + " " +
    movies["description"] + " " +
    movies["keywords"]
).apply(clean_text)

movies[["title", "combined_text"]].head()
"""),
        md("Bag-of-Words counts words. Common words can dominate because each word is weighted mostly by frequency."),
        code(r"""
count_vectorizer = CountVectorizer(stop_words="english")
bow_matrix = count_vectorizer.fit_transform(movies["combined_text"])

bow_preview = pd.DataFrame(
    bow_matrix.toarray(),
    columns=count_vectorizer.get_feature_names_out(),
    index=movies["title"],
)
bow_preview.iloc[:5, :12]
"""),
        md("TF-IDF lowers the weight of terms that appear in many movies and raises distinctive terms."),
        code(r"""
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(movies["combined_text"])

tfidf_preview = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out(),
    index=movies["title"],
)
tfidf_preview.iloc[:5, :12].round(2)
"""),
        md("The similarity matrix compares every movie with every other movie."),
        code(r"""
bow_similarity = cosine_similarity(bow_matrix)
tfidf_similarity = cosine_similarity(tfidf_matrix)

pd.DataFrame(tfidf_similarity, index=movies["title"], columns=movies["title"]).round(2)
"""),
        md("This function returns Top-N similar movies and shows which TF-IDF terms are shared with the input movie."),
        code(r"""
def shared_terms(input_idx, other_idx, matrix, vectorizer, top_terms=6):
    feature_names = np.array(vectorizer.get_feature_names_out())
    input_weights = matrix[input_idx].toarray().ravel()
    other_weights = matrix[other_idx].toarray().ravel()
    shared = np.minimum(input_weights, other_weights)
    best = shared.argsort()[::-1][:top_terms]
    return ", ".join(feature_names[i] for i in best if shared[i] > 0)

def recommend_similar(title, similarity_matrix, matrix, vectorizer, n=5):
    idx = movies.index[movies["title"].eq(title)][0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    rows = []
    for other_idx, score in scores[1:n+1]:
        rows.append({
            "input_movie": title,
            "recommended_movie": movies.loc[other_idx, "title"],
            "similarity_score": round(float(score), 3),
            "shared_terms_features": shared_terms(idx, other_idx, matrix, vectorizer),
        })
    return pd.DataFrame(rows)

recommend_similar("Interstellar", tfidf_similarity, tfidf_matrix, tfidf_vectorizer)
"""),
        md("Compare Bag-of-Words and TF-IDF. The rankings may be similar, but TF-IDF usually gives cleaner emphasis to distinctive content."),
        code(r"""
bow_results = recommend_similar("Interstellar", bow_similarity, bow_matrix, count_vectorizer, n=5)
tfidf_results = recommend_similar("Interstellar", tfidf_similarity, tfidf_matrix, tfidf_vectorizer, n=5)

comparison = bow_results[["recommended_movie", "similarity_score"]].rename(columns={"similarity_score": "bow_score"})
comparison["tfidf_movie"] = tfidf_results["recommended_movie"]
comparison["tfidf_score"] = tfidf_results["similarity_score"]
comparison
"""),
        md("""
## What did we learn?

- Bag-of-Words creates count vectors from text.
- TF-IDF keeps the vector idea but gives more weight to distinctive terms.
- Cosine similarity turns text vectors into a ranked recommendation list.

Exercises:
1. Try the recommender with `Toy Story` or `Titanic`.
2. Add a new keyword to one movie and check whether the ranking changes.
"""),
    ]


def notebook_03() -> list[dict]:
    return [
        md("""
# Chapter 2 Practical 03: User Profile and Ranking

Learning objectives:
- Build a user profile from liked movies.
- Compare average, rating-weighted, normalized, implicit-feedback, and temporal-decay profiles.
- Recommend unseen movies.
- Explain recommendations using overlapping features.

Slide connection: user profiles, similarity matching, ranking, and Top-N recommendation.
"""),
        md("Load movie metadata and a small set of user interactions."),
        code(COMMON_LOAD + r"""
interactions = pd.read_csv(DATA_DIR / "user_interactions_chapter2.csv")
interactions.head()
"""),
        md("Create an item-feature matrix from genres, directors, and text. This gives the user profile a mix of structured and textual evidence."),
        code(r"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

movies["feature_text"] = (
    movies["genres"].str.replace("|", " ", regex=False) + " " +
    movies["director"] + " " +
    movies["description"] + " " +
    movies["keywords"]
)
tfidf = TfidfVectorizer(stop_words="english")
text_features = tfidf.fit_transform(movies["feature_text"]).toarray()

numeric = MinMaxScaler().fit_transform(movies[["duration_min", "rating", "family_friendly"]])
feature_names = list(tfidf.get_feature_names_out()) + ["duration_scaled", "rating_scaled", "family_friendly"]
item_features = pd.DataFrame(
    np.hstack([text_features, numeric]),
    index=movies["title"],
    columns=feature_names,
)
item_features.iloc[:5, -8:].round(2)
"""),
        md("For a simple average profile, each liked item has the same influence."),
        code(r"""
user_id = "U1"
user_history = interactions[interactions["user_id"] == user_id].copy()
liked_titles = user_history["title"].tolist()

average_profile = item_features.loc[liked_titles].mean(axis=0)
average_profile.sort_values(ascending=False).head(10).round(3)
"""),
        md("A rating-weighted profile gives stronger liked items more influence."),
        code(r"""
weights_rating = user_history.set_index("title")["rating"]
rating_weighted_profile = item_features.loc[liked_titles].mul(weights_rating, axis=0).sum() / weights_rating.sum()
rating_weighted_profile.sort_values(ascending=False).head(10).round(3)
"""),
        md("A rating-normalized profile centers ratings around the user's average. This reduces the effect of users who rate everything high."),
        code(r"""
normalized_weights = weights_rating - weights_rating.mean()
if normalized_weights.abs().sum() == 0:
    normalized_weights = weights_rating / weights_rating.sum()

rating_normalized_profile = item_features.loc[liked_titles].mul(normalized_weights, axis=0).sum()
rating_normalized_profile.sort_values(ascending=False).head(10).round(3)
"""),
        md("Implicit feedback can combine clicks, likes, and watch time. Temporal decay gives recent interactions more weight."),
        code(r"""
max_duration = movies.set_index("title")["duration_min"]
history = user_history.set_index("title")
watch_ratio = history["watch_minutes"] / max_duration.loc[history.index]
implicit_weight = 0.2 * history["clicked"] + 0.5 * history["liked"] + 0.3 * watch_ratio
temporal_decay = np.exp(-history["days_ago"] / 30)
final_weight = implicit_weight * temporal_decay

profile_temporal = item_features.loc[history.index].mul(final_weight, axis=0).sum() / final_weight.sum()
pd.DataFrame({
    "implicit_weight": implicit_weight.round(3),
    "temporal_decay": temporal_decay.round(3),
    "final_weight": final_weight.round(3),
})
"""),
        md("Recommend unseen movies by comparing each item vector to the user profile."),
        code(r"""
def recommend_from_profile(profile, seen_titles, top_n=5):
    candidate_features = item_features.drop(index=seen_titles)
    scores = cosine_similarity(candidate_features, profile.values.reshape(1, -1)).ravel()
    results = pd.DataFrame({"title": candidate_features.index, "score": scores})
    return results.sort_values("score", ascending=False).head(top_n)

recommendations = recommend_from_profile(profile_temporal, liked_titles)
recommendations.round(3)
"""),
        md("We can explain each recommendation by showing the strongest features shared by the user profile and the movie."),
        code(r"""
def explain_recommendation(title, profile, top_n=6):
    contribution = item_features.loc[title] * profile
    return contribution.sort_values(ascending=False).head(top_n).round(3)

best_title = recommendations.iloc[0]["title"]
print(f"Explanation for {best_title}:")
explain_recommendation(best_title, profile_temporal)
"""),
        md("""
## What did we learn?

- A user profile is a vector summarizing what the user liked.
- Different weighting choices create different profiles.
- Recommendations become more transparent when we inspect shared high-weight features.

Exercises:
1. Change `user_id` to `U2` or `U3` and compare the profile terms.
2. Increase or decrease the temporal decay speed. Which recommendations change?
"""),
    ]


def notebook_04() -> list[dict]:
    return [
        md("""
# Chapter 2 Practical 04: Evaluation of Top-K Recommendations

Learning objectives:
- Compute Precision@K, Recall@K, and HitRate@K.
- Evaluate recommendations for multiple users.
- Read a compact metric table and bar chart.
- Distinguish accuracy, coverage, and hit-oriented satisfaction.

Slide connection: Precision@K, Recall@K, HitRate@K, ranking, and Top-N evaluation.
"""),
        md("We use a tiny ground-truth example. In a real project, relevant items usually come from held-out ratings, clicks, purchases, or watch events."),
        code(r"""
import pandas as pd
import matplotlib.pyplot as plt

ground_truth = {
    "U1": {"The Martian", "Gravity", "The Matrix"},
    "U2": {"Toy Story", "Finding Nemo", "Paddington"},
    "U3": {"Titanic", "The Notebook", "La La Land"},
}

recommendations = {
    "U1": ["The Martian", "The Dark Knight", "Gravity", "Titanic", "Paddington"],
    "U2": ["Paddington", "Toy Story", "The Martian", "Finding Nemo", "Inception"],
    "U3": ["La La Land", "Titanic", "Interstellar", "The Notebook", "Toy Story"],
}

pd.DataFrame([
    {"user": u, "ground_truth": sorted(gt), "recommendations": recommendations[u]}
    for u, gt in ground_truth.items()
])
"""),
        md("Precision@K asks: of the K recommended items, how many were relevant?"),
        code(r"""
def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / k

def recall_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(relevant) if relevant else 0

def hitrate_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    return int(len(set(recommended_k) & set(relevant)) > 0)

precision_at_k(recommendations["U1"], ground_truth["U1"], k=3)
"""),
        md("Now evaluate all users and display the results in one clean table."),
        code(r"""
k = 3
rows = []
for user, relevant in ground_truth.items():
    recs = recommendations[user]
    rows.append({
        "user": user,
        f"Precision@{k}": precision_at_k(recs, relevant, k),
        f"Recall@{k}": recall_at_k(recs, relevant, k),
        f"HitRate@{k}": hitrate_at_k(recs, relevant, k),
        "hits_in_top_k": sorted(set(recs[:k]) & relevant),
    })

metrics = pd.DataFrame(rows)
metrics
"""),
        md("Average the metrics across users to summarize system performance."),
        code(r"""
summary = metrics[[f"Precision@{k}", f"Recall@{k}", f"HitRate@{k}"]].mean().to_frame("mean_value")
summary.round(3)
"""),
        md("A simple bar chart makes it easier to compare the metric values."),
        code(r"""
ax = summary.plot(kind="bar", legend=False, ylim=(0, 1), figsize=(6, 4))
ax.set_ylabel("Metric value")
ax.set_title(f"Average Top-{k} evaluation")
ax.bar_label(ax.containers[0], fmt="%.2f")
plt.xticks(rotation=0)
plt.show()
"""),
        md("""
Accuracy, coverage, and satisfaction are related but not identical:

- Accuracy asks whether recommended items match known relevant items.
- Coverage asks whether the system can recommend a broad set of items instead of always the same few.
- HitRate@K asks whether the list contains at least one useful item for the user.

## What did we learn?

- Precision@K rewards short lists with many relevant items.
- Recall@K rewards finding a large share of all relevant items.
- HitRate@K is forgiving: one hit is enough.

Exercises:
1. Change `k` from 3 to 5. Which metric changes the most?
2. Add one more user with recommendations and ground truth.
"""),
    ]


def notebook_05() -> list[dict]:
    return [
        md("""
# Chapter 2 Practical 05: SBERT Semantic Recommender

Learning objectives:
- Explain lexical similarity versus semantic similarity.
- Encode movie descriptions with SBERT when available.
- Fall back to TF-IDF when `sentence-transformers` or the model is unavailable.
- Compare TF-IDF and semantic recommendation results.

Slide connection: deep content models, SBERT/BERT embeddings, semantic similarity, and practical fallback design.
"""),
        md("TF-IDF works with shared words. SBERT can also capture related meanings, such as `astronaut`, `space`, `orbit`, and `Mars`."),
        code(COMMON_LOAD),
        md("First build a TF-IDF baseline that always works in a basic Python environment."),
        code(r"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies["semantic_text"] = movies["title"] + ". " + movies["description"] + " Keywords: " + movies["keywords"]

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["semantic_text"])
tfidf_similarity = cosine_similarity(tfidf_matrix)
"""),
        md("Now try SBERT. If the package is missing or the model cannot be downloaded, the notebook continues with the TF-IDF fallback."),
        code(r"""
embedding_source = "tfidf fallback"
sbert_similarity = tfidf_similarity

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(movies["semantic_text"].tolist(), show_progress_bar=False)
    sbert_similarity = cosine_similarity(embeddings)
    embedding_source = "sentence-transformers/all-MiniLM-L6-v2"
except Exception as exc:
    print("SBERT is not available in this environment.")
    print("Using TF-IDF fallback instead.")
    print(type(exc).__name__, str(exc)[:160])

embedding_source
"""),
        md("Use the same recommendation function for either similarity matrix."),
        code(r"""
def recommend(title, similarity_matrix, n=5):
    idx = movies.index[movies["title"].eq(title)][0]
    scores = sorted(enumerate(similarity_matrix[idx]), key=lambda x: x[1], reverse=True)
    return pd.DataFrame([
        {"query_movie": title, "recommended_movie": movies.loc[i, "title"], "score": round(float(score), 3)}
        for i, score in scores[1:n+1]
    ])

recommend("Interstellar", sbert_similarity)
"""),
        md("Compare lexical TF-IDF results with semantic SBERT results. If SBERT is not available, both columns will show the fallback behavior."),
        code(r"""
tfidf_results = recommend("Interstellar", tfidf_similarity, n=5).rename(columns={
    "recommended_movie": "tfidf_recommendation",
    "score": "tfidf_score",
})
sbert_results = recommend("Interstellar", sbert_similarity, n=5).rename(columns={
    "recommended_movie": "semantic_recommendation",
    "score": "semantic_score",
})

pd.concat([
    tfidf_results[["tfidf_recommendation", "tfidf_score"]],
    sbert_results[["semantic_recommendation", "semantic_score"]],
], axis=1)
"""),
        md("Zero-shot style semantic search uses a text query instead of an input movie."),
        code(r"""
queries = ["movies about space exploration", "romantic drama about lifelong love"]

if embedding_source.startswith("sentence-transformers"):
    query_embeddings = model.encode(queries, show_progress_bar=False)
    query_scores = cosine_similarity(query_embeddings, embeddings)
else:
    query_matrix = tfidf.transform(queries)
    query_scores = cosine_similarity(query_matrix, tfidf_matrix)

rows = []
for q_idx, query in enumerate(queries):
    best = query_scores[q_idx].argsort()[::-1][:4]
    for movie_idx in best:
        rows.append({"query": query, "movie": movies.loc[movie_idx, "title"], "score": round(float(query_scores[q_idx, movie_idx]), 3)})

pd.DataFrame(rows)
"""),
        md("""
## What did we learn?

- TF-IDF is lexical: shared words drive similarity.
- SBERT is semantic: related meanings can be close even with different words.
- Optional models should have fallback logic so teaching notebooks still run.

Exercises:
1. Try the query `mind-bending action movie`.
2. Add a new movie description that avoids the word `space` but is clearly about astronauts.
"""),
    ]


def notebook_06() -> list[dict]:
    return [
        md("""
# Chapter 2 Practical 06: Graph-Based Content Recommendation

Learning objectives:
- Build a movie graph from shared content features.
- Visualize movies as nodes and content relationships as edges.
- Compute graph-based similarity using shared neighbors.
- Understand the message-passing idea in simple terms.

Slide connection: graph-based content recommendation, item relationships, graph neighborhoods, and optional graph embeddings.
"""),
        md("A content graph connects items when they share features such as genres, directors, or high text similarity."),
        code(COMMON_LOAD),
        md("Build a graph where nodes are movies and edges mean shared genre or same director."),
        code(r"""
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

G = nx.Graph()
for _, row in movies.iterrows():
    G.add_node(row["title"], genres=set(row["genres"].split("|")), director=row["director"], rating=row["rating"])

for a, b in combinations(movies["title"], 2):
    a_data, b_data = G.nodes[a], G.nodes[b]
    shared_genres = a_data["genres"] & b_data["genres"]
    same_director = a_data["director"] == b_data["director"]
    if shared_genres or same_director:
        weight = len(shared_genres) + (1.5 if same_director else 0)
        reason = ", ".join(sorted(shared_genres))
        if same_director:
            reason = (reason + ", " if reason else "") + "same director"
        G.add_edge(a, b, weight=weight, reason=reason)

print("nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())
list(G.edges(data=True))[:8]
"""),
        md("Visualize the graph. Thicker edges mean stronger content overlap."),
        code(r"""
plt.figure(figsize=(9, 6))
pos = nx.spring_layout(G, seed=7)
edge_widths = [G[u][v]["weight"] for u, v in G.edges()]
nx.draw_networkx_nodes(G, pos, node_size=900, node_color="#dbeafe", edgecolors="#1f2937")
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.45)
nx.draw_networkx_labels(G, pos, font_size=8)
plt.title("Movie content graph")
plt.axis("off")
plt.show()
"""),
        md("A simple graph similarity score can combine direct edge strength and shared neighbors."),
        code(r"""
def graph_similarity(source, target):
    direct = G[source][target]["weight"] if G.has_edge(source, target) else 0
    common = len(list(nx.common_neighbors(G, source, target)))
    return direct + 0.5 * common

def graph_recommend(title, n=5):
    rows = []
    for other in G.nodes:
        if other == title:
            continue
        rows.append({
            "input_movie": title,
            "recommended_movie": other,
            "graph_score": graph_similarity(title, other),
            "edge_reason": G[title][other]["reason"] if G.has_edge(title, other) else "shared neighbors only",
        })
    return pd.DataFrame(rows).sort_values("graph_score", ascending=False).head(n)

graph_recommend("Interstellar")
"""),
        md("Centrality can identify well-connected items. This is not personalization by itself, but it helps explain graph structure."),
        code(r"""
centrality = pd.Series(nx.degree_centrality(G), name="degree_centrality").sort_values(ascending=False)
centrality.head(8).round(3)
"""),
        md("Optional node embeddings can be useful, but the notebook remains runnable without them."),
        code(r"""
try:
    from node2vec import Node2Vec
    node2vec = Node2Vec(G, dimensions=8, walk_length=5, num_walks=30, workers=1, quiet=True, seed=7)
    model = node2vec.fit(window=3, min_count=1, batch_words=16)
    print("Node2Vec embeddings learned.")
    print(model.wv.most_similar("Interstellar", topn=5))
except Exception as exc:
    print("Optional Node2Vec is not available. The NetworkX graph recommendation above is the fallback.")
    print(type(exc).__name__, str(exc)[:160])
"""),
        md("""
Message passing idea:

Each movie can receive information from its neighbors. After one step, a movie knows about directly connected movies. After two steps, it also receives signals from neighbors of neighbors. Graph neural networks formalize this idea with learned transformations.

## What did we learn?

- Graphs represent relationships between items, not only feature rows.
- Shared genres and directors can create interpretable edges.
- A simple NetworkX fallback is enough to teach graph-based recommendation before advanced GNNs.

Exercises:
1. Add edges for high TF-IDF similarity and compare the graph.
2. Recommend movies from a two-item user profile by averaging graph scores.
"""),
    ]


def notebook_07() -> list[dict]:
    return [
        md("""
# Chapter 2 Practical 07: Context, Explainability, and Zero-Shot Demo

Learning objectives:
- Apply context-aware re-ranking.
- Explain recommendations with shared content features.
- Build a zero-shot style text search interface.
- Keep generative enrichment as an optional stub, not a required API call.

Slide connection: context-aware recommendation, explainable recommendation, zero-shot examples, and generative metadata enrichment.
"""),
        md("Load the same movie data so this notebook connects back to the earlier practicals."),
        code(COMMON_LOAD),
        md("Create a base TF-IDF recommender. This acts as the reliable fallback for zero-shot text queries."),
        code(r"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies["search_text"] = (
    movies["title"] + " " +
    movies["genres"].str.replace("|", " ", regex=False) + " " +
    movies["director"] + " " +
    movies["description"] + " " +
    movies["keywords"]
)

vectorizer = TfidfVectorizer(stop_words="english")
item_matrix = vectorizer.fit_transform(movies["search_text"])
"""),
        md("Zero-shot style search lets the user describe what they want instead of choosing a seed item."),
        code(r"""
def search_movies(query, n=6):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, item_matrix).ravel()
    results = movies[["title", "genres", "director", "family_friendly", "duration_min"]].copy()
    results["base_score"] = scores
    return results.sort_values("base_score", ascending=False).head(n)

search_movies("movies about space exploration")
"""),
        md("Context-aware recommendation adjusts the ranking for the current situation."),
        code(r"""
def rerank_for_context(results, context):
    adjusted = results.copy()
    adjusted["context_bonus"] = 0.0

    if context == "morning_mobile":
        adjusted.loc[adjusted["duration_min"] <= 110, "context_bonus"] += 0.12
    elif context == "evening_tv":
        adjusted.loc[adjusted["duration_min"] >= 120, "context_bonus"] += 0.10
    elif context == "family_mode":
        adjusted.loc[adjusted["family_friendly"] == 1, "context_bonus"] += 0.20

    adjusted["final_score"] = adjusted["base_score"] + adjusted["context_bonus"]
    return adjusted.sort_values("final_score", ascending=False)

base = search_movies("light comedy for family evening", n=8)
rerank_for_context(base, "family_mode")
"""),
        md("Explanations should be short and specific. Here we explain by shared genres, director, and high-weight query terms."),
        code(r"""
def explain_with_features(query, title, top_terms=5):
    movie = movies[movies["title"].eq(title)].iloc[0]
    query_vector = vectorizer.transform([query]).toarray().ravel()
    movie_vector = item_matrix[movies.index[movies["title"].eq(title)][0]].toarray().ravel()
    contribution = query_vector * movie_vector
    terms = vectorizer.get_feature_names_out()
    best_terms = [terms[i] for i in contribution.argsort()[::-1][:top_terms] if contribution[i] > 0]

    return {
        "movie": title,
        "recommended_because_it_shares": ", ".join(best_terms) if best_terms else "related content features",
        "genres": movie["genres"],
        "director": movie["director"],
    }

query = "space survival astronaut"
top_title = search_movies(query, n=1).iloc[0]["title"]
explain_with_features(query, top_title)
"""),
        md("A contribution table makes the explanation inspectable rather than magical."),
        code(r"""
def contribution_table(query, title):
    idx = movies.index[movies["title"].eq(title)][0]
    q = vectorizer.transform([query]).toarray().ravel()
    x = item_matrix[idx].toarray().ravel()
    terms = vectorizer.get_feature_names_out()
    table = pd.DataFrame({"term": terms, "query_weight": q, "movie_weight": x})
    table["contribution"] = table["query_weight"] * table["movie_weight"]
    return table[table["contribution"] > 0].sort_values("contribution", ascending=False).head(10)

contribution_table("space survival astronaut", top_title)
"""),
        md("Optional SBERT can replace TF-IDF for zero-shot search if it is available."),
        code(r"""
semantic_search_available = False
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(movies["search_text"].tolist(), show_progress_bar=False)
    semantic_search_available = True
except Exception as exc:
    print("Semantic search model is optional and not available here. TF-IDF search remains active.")
    print(type(exc).__name__, str(exc)[:160])

semantic_search_available
"""),
        md("Generative enrichment can be introduced as a stub. Students should not need an API key to run the notebook."),
        code(r"""
def enrich_metadata_stub(title):
    return {
        "title": title,
        "possible_extra_tags": ["teaching stub", "replace with reviewed metadata", "no API call required"],
        "note": "In production, generated metadata should be checked before it affects recommendations.",
    }

enrich_metadata_stub("Interstellar")
"""),
        md("""
## What did we learn?

- Context can re-rank otherwise reasonable recommendations.
- Explanations should name concrete shared features.
- Zero-shot search can be taught with TF-IDF first and upgraded to embeddings when available.
- Generative enrichment is powerful, but it should be optional and reviewed.

Exercises:
1. Add a `late_night` context and define your own re-ranking rule.
2. Write two natural-language queries and compare their recommendation lists.
"""),
    ]


def slide_exercise_01_tfidf() -> list[dict]:
    return [
        md("""
# Slide Exercise 01: Expanded TF-IDF Movie Recommender

This is the refined version of `TFIDF_MovieRecommender_Expanded.ipynb`.

Learning objectives:
- Build a stronger TF-IDF content representation from title, genres, director, description, and keywords.
- Generate Top-N similar movies.
- Explain recommendations with shared weighted terms.

Main functions used:
- `TfidfVectorizer(...)`: converts text into weighted term vectors.
- `fit_transform(...)`: learns the vocabulary and creates the movie-term matrix.
- `cosine_similarity(...)`: compares movie vectors by angle.
- `argsort()`: sorts similarity scores to create a ranking.
"""),
        md("Load the shared Chapter 2 dataset. We keep the dataset small so students can inspect every intermediate result."),
        code(COMMON_LOAD),
        md("Create one clean text field per movie. In real systems this step is often called metadata fusion."),
        code(r"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

movies["content"] = (
    movies["title"] + " " +
    movies["genres"].str.replace("|", " ", regex=False) + " " +
    movies["director"] + " " +
    movies["description"] + " " +
    movies["keywords"]
).apply(clean_text)

movies[["title", "content"]].head()
"""),
        md("Fit TF-IDF and compute a movie-by-movie similarity matrix."),
        code(r"""
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
tfidf_matrix = vectorizer.fit_transform(movies["content"])
similarity = cosine_similarity(tfidf_matrix)

print("TF-IDF matrix shape:", tfidf_matrix.shape)
pd.DataFrame(similarity, index=movies["title"], columns=movies["title"]).round(2)
"""),
        md("Define small reusable functions. The recommendation function ranks movies; the explanation function shows shared TF-IDF features."),
        code(r"""
def shared_weighted_terms(seed_idx, other_idx, top_n=6):
    terms = np.array(vectorizer.get_feature_names_out())
    seed_weights = tfidf_matrix[seed_idx].toarray().ravel()
    other_weights = tfidf_matrix[other_idx].toarray().ravel()
    shared_weight = np.minimum(seed_weights, other_weights)
    best = shared_weight.argsort()[::-1][:top_n]
    return ", ".join(terms[i] for i in best if shared_weight[i] > 0)

def recommend_similar_movies(title, n=5):
    seed_idx = movies.index[movies["title"].eq(title)][0]
    ranked = similarity[seed_idx].argsort()[::-1]
    rows = []
    for other_idx in ranked:
        if other_idx == seed_idx:
            continue
        rows.append({
            "input_movie": title,
            "recommended_movie": movies.loc[other_idx, "title"],
            "similarity": round(float(similarity[seed_idx, other_idx]), 3),
            "shared_terms": shared_weighted_terms(seed_idx, other_idx),
        })
        if len(rows) == n:
            break
    return pd.DataFrame(rows)

recommend_similar_movies("Interstellar")
"""),
        md("""
Interpretation:

Movies with shared terms such as `space`, `astronaut`, `sci-fi`, or a shared director move upward in the ranking. TF-IDF is still lexical, so it works best when related movies use overlapping vocabulary.

Student task:
1. Change the input movie to `Toy Story`.
2. Remove bigrams by setting `ngram_range=(1, 1)`. Did the ranking change?
"""),
    ]


def slide_exercise_02_sbert() -> list[dict]:
    return [
        md("""
# Slide Exercise 02: SBERT Semantic Movie Recommender

This is the refined version of `SBERT_MovieRecommender.ipynb`.

Learning objectives:
- Compare lexical TF-IDF similarity with semantic embedding similarity.
- Use `all-MiniLM-L6-v2` when available.
- Keep the exercise runnable with a TF-IDF fallback.

Main functions used:
- `SentenceTransformer(...)`: loads a pretrained sentence embedding model.
- `model.encode(...)`: converts descriptions into dense semantic vectors.
- `cosine_similarity(...)`: compares embedding vectors.
- `try/except`: keeps optional model code from breaking the notebook.
"""),
        code(COMMON_LOAD),
        md("Build a text field that reads like a short movie profile."),
        code(r"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

movies["profile_text"] = movies["title"] + ". " + movies["description"] + " Genres: " + movies["genres"].str.replace("|", ", ", regex=False)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["profile_text"])
tfidf_similarity = cosine_similarity(tfidf_matrix)
"""),
        md("Try SBERT. If it is missing or cannot load the model, use the TF-IDF matrix instead."),
        code(r"""
model_name = "TF-IDF fallback"
semantic_similarity = tfidf_similarity
semantic_vectors = tfidf_matrix

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    semantic_vectors = model.encode(movies["profile_text"].tolist(), show_progress_bar=False)
    semantic_similarity = cosine_similarity(semantic_vectors)
    model_name = "SBERT all-MiniLM-L6-v2"
except Exception as exc:
    print("SBERT is optional for this exercise. Continuing with TF-IDF fallback.")
    print(type(exc).__name__, str(exc)[:160])

model_name
"""),
        md("Use one ranking function for both lexical and semantic similarities."),
        code(r"""
def recommend(title, similarity_matrix, label, n=5):
    idx = movies.index[movies["title"].eq(title)][0]
    ranked = similarity_matrix[idx].argsort()[::-1]
    rows = []
    for other_idx in ranked:
        if other_idx == idx:
            continue
        rows.append({
            "method": label,
            "input_movie": title,
            "recommended_movie": movies.loc[other_idx, "title"],
            "score": round(float(similarity_matrix[idx, other_idx]), 3),
        })
        if len(rows) == n:
            break
    return pd.DataFrame(rows)

pd.concat([
    recommend("Gravity", tfidf_similarity, "TF-IDF"),
    recommend("Gravity", semantic_similarity, model_name),
], ignore_index=True)
"""),
        md("""
Interpretation:

TF-IDF rewards shared words. SBERT, when available, can connect descriptions that express similar meaning with different words.

Student task:
1. Compare recommendations for `Titanic`.
2. Add a new movie with a description that uses different words for a similar idea.
"""),
    ]


def slide_exercise_03_graph() -> list[dict]:
    return [
        md("""
# Slide Exercise 03: Graph-Based Movie Recommender

This is the refined version of `GraphCB_MovieRecommender_NodeVectors.ipynb`.

Learning objectives:
- Represent movies as graph nodes.
- Connect movies by shared genres and shared director.
- Recommend from graph neighborhoods.
- Understand where optional node embeddings fit.

Main functions used:
- `nx.Graph()`: creates an undirected graph.
- `add_node(...)` and `add_edge(...)`: add movies and content relationships.
- `nx.spring_layout(...)`: computes node positions for visualization.
- `nx.common_neighbors(...)`: finds graph-neighborhood overlap.
"""),
        code(COMMON_LOAD),
        md("Create a graph. Edge weights increase when movies share more content features."),
        code(r"""
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd

G = nx.Graph()
for _, movie in movies.iterrows():
    G.add_node(movie["title"], genres=set(movie["genres"].split("|")), director=movie["director"])

for a, b in combinations(movies["title"], 2):
    a_data = G.nodes[a]
    b_data = G.nodes[b]
    shared_genres = a_data["genres"] & b_data["genres"]
    same_director = a_data["director"] == b_data["director"]
    if shared_genres or same_director:
        G.add_edge(
            a,
            b,
            weight=len(shared_genres) + (1.5 if same_director else 0),
            reason=", ".join(sorted(shared_genres)) + ("; same director" if same_director else ""),
        )

print("Movies:", G.number_of_nodes(), "Content edges:", G.number_of_edges())
"""),
        md("Visualize the content graph."),
        code(r"""
plt.figure(figsize=(9, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_size=900, node_color="#ecfeff", edgecolors="#334155")
nx.draw_networkx_edges(G, pos, width=[G[u][v]["weight"] for u, v in G.edges()], alpha=0.45)
nx.draw_networkx_labels(G, pos, font_size=8)
plt.title("Content graph from shared genres and director")
plt.axis("off")
plt.show()
"""),
        md("Recommend using direct graph strength plus common-neighbor overlap."),
        code(r"""
def graph_score(source, target):
    direct = G[source][target]["weight"] if G.has_edge(source, target) else 0
    neighbor_overlap = len(list(nx.common_neighbors(G, source, target)))
    return direct + 0.5 * neighbor_overlap

def recommend_from_graph(title, n=5):
    rows = []
    for other in G.nodes:
        if other == title:
            continue
        rows.append({
            "input_movie": title,
            "recommended_movie": other,
            "graph_score": graph_score(title, other),
            "edge_reason": G[title][other]["reason"] if G.has_edge(title, other) else "shared graph neighborhood",
        })
    return pd.DataFrame(rows).sort_values("graph_score", ascending=False).head(n)

recommend_from_graph("Interstellar")
"""),
        md("""
Optional extension:

Node2Vec or GraphSAGE can learn node embeddings from graph neighborhoods. For this course exercise, the NetworkX version is the required path because it is transparent and dependency-light.

Student task:
1. Add an edge rule for similar duration.
2. Compare graph recommendations for `Toy Story` before and after the new rule.
"""),
    ]


def slide_exercise_04_multimodal() -> list[dict]:
    return [
        md("""
# Slide Exercise 04: Multi-Modal Movie Recommender

This is the refined version of `MultiModal_MovieRecommender.ipynb`.

Learning objectives:
- Display real movie posters so the recommendation output is visible, not only numerical.
- Combine text features with poster/image features.
- Compare text-only and multi-modal rankings.
- Use optional CLIP image embeddings when available, with a simple fallback.

Main functions used:
- `TfidfVectorizer(...)`: creates text vectors.
- `requests.get(...)`: downloads poster images from URLs.
- `Image.open(...)`: opens downloaded poster images with Pillow.
- `plt.imshow(...)`: displays posters in the notebook.
- `SentenceTransformer('clip-ViT-B-32')`: optionally encodes poster images into image embeddings.
- `np.hstack(...)`: combines text and image/visual feature arrays.
- `cosine_similarity(...)`: ranks movies after feature fusion.
"""),
        md("Use a small MovieLens-style subset with real poster URLs. This mirrors the earlier notebook you liked, but adds safer fallback logic and clearer teaching explanations."),
        code(r"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display, HTML
from io import BytesIO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests

movies = pd.DataFrame({
    "movieId": [1, 2, 3, 4, 5, 6],
    "title": [
        "Toy Story",
        "Jumanji",
        "Grumpier Old Men",
        "Waiting to Exhale",
        "Father of the Bride Part II",
        "Heat",
    ],
    "genres": [
        "Animation|Children|Comedy",
        "Adventure|Children|Fantasy",
        "Comedy|Romance",
        "Comedy|Drama|Romance",
        "Comedy",
        "Action|Crime|Thriller",
    ],
    "poster_url": [
        "https://image.tmdb.org/t/p/w500/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg",
        "https://image.tmdb.org/t/p/w500/vzmL6fP7aPKNKPRTFnZmiUfciyV.jpg",
        "https://image.tmdb.org/t/p/w500/qvktm0BHcnmDpul4Hz01GIazWPr.jpg",
        "https://image.tmdb.org/t/p/w500/sJnO6G3HXHzeT6rv2yzZbR6quT3.jpg",
        "https://image.tmdb.org/t/p/w500/3X0XKxLZ6fHf8mJ7gu9BID8K0F6.jpg",
        "https://image.tmdb.org/t/p/w500/rr7E0NoGKxvbkb89eR1GwfoYjpA.jpg",
    ],
})

movies
"""),
        md("Display the posters directly in the notebook. Even before embeddings, students can see that multi-modal recommendation means using more than text."),
        code(r"""
def show_poster_grid(movie_table, title="Movie posters"):
    cards = []
    for _, row in movie_table.iterrows():
        cards.append(
            f'''
            <div style="width:150px; margin:8px; display:inline-block; vertical-align:top; text-align:center;">
              <img src="{row['poster_url']}" style="width:140px; height:210px; object-fit:cover; border-radius:6px;">
              <div style="font-size:13px; margin-top:6px;">{row['title']}</div>
            </div>
            '''
        )
    display(HTML(f"<h3>{title}</h3>" + "".join(cards)))

show_poster_grid(movies)
"""),
        md("Create text vectors from title and genres. This is the text modality."),
        code(r"""
movies["text"] = movies["title"] + " " + movies["genres"].str.replace("|", " ", regex=False)

tfidf = TfidfVectorizer(stop_words="english")
text_matrix = tfidf.fit_transform(movies["text"]).toarray()
text_similarity = cosine_similarity(text_matrix)

pd.DataFrame(text_matrix, index=movies["title"], columns=tfidf.get_feature_names_out()).round(2)
"""),
        md("Try to load real poster images. If the network is unavailable during class, the function returns a simple placeholder image instead of breaking the notebook."),
        code(r"""
def placeholder_image(title, size=(220, 330)):
    image = Image.new("RGB", size, color=(235, 238, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, size[0] - 1, size[1] - 1], outline=(120, 130, 150), width=3)
    draw.text((14, 20), title[:22], fill=(30, 40, 60))
    draw.text((14, 55), "poster unavailable", fill=(80, 90, 110))
    return image

def load_poster(url, title, timeout=8):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as exc:
        print(f"Using placeholder for {title}: {type(exc).__name__}")
        return placeholder_image(title)

poster_images = [
    load_poster(row.poster_url, row.title)
    for row in movies.itertuples(index=False)
]

print("Loaded poster images:", len(poster_images))
"""),
        md("Optional path: encode real poster images with CLIP. Fallback path: use transparent visual features extracted from the poster pixels, such as average color and brightness."),
        code(r"""
def normalize_rows(matrix):
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms

embedding_source = "simple poster pixel features"

try:
    from sentence_transformers import SentenceTransformer
    image_model = SentenceTransformer("clip-ViT-B-32")
    image_matrix = image_model.encode(poster_images, show_progress_bar=False)
    embedding_source = "CLIP poster embeddings"
except Exception as exc:
    print("CLIP image embeddings are optional. Using simple poster pixel features instead.")
    print(type(exc).__name__, str(exc)[:160])
    rows = []
    for image in poster_images:
        small = image.resize((1, 1))
        r, g, b = np.array(small)[0, 0] / 255
        gray = image.convert("L").resize((1, 1))
        brightness = np.array(gray)[0, 0] / 255
        rows.append([r, g, b, brightness])
    image_matrix = StandardScaler().fit_transform(np.array(rows))

text_weight = 0.65
image_weight = 0.35
multimodal_matrix = np.hstack([
    text_weight * normalize_rows(text_matrix),
    image_weight * normalize_rows(image_matrix),
])

multimodal_similarity = cosine_similarity(multimodal_matrix)
embedding_source
"""),
        md("Compare rankings. Multi-modal fusion can slightly change the order when visual style supports or weakens the text match."),
        code(r"""
def recommend(title, similarity_matrix, label, n=5):
    idx = movies.index[movies["title"].eq(title)][0]
    ranked = similarity_matrix[idx].argsort()[::-1]
    rows = []
    for other_idx in ranked:
        if other_idx == idx:
            continue
        rows.append({
            "method": label,
            "input_movie": title,
            "recommended_movie": movies.loc[other_idx, "title"],
            "score": round(float(similarity_matrix[idx, other_idx]), 3),
        })
        if len(rows) == n:
            break
    return pd.DataFrame(rows)

pd.concat([
    recommend("Toy Story", text_similarity, "text only"),
    recommend("Toy Story", multimodal_similarity, "text + posters"),
], ignore_index=True)
"""),
        md("Visualize the query movie and its recommendations. This is the main classroom payoff: students see the recommended items, not only a score table."),
        code(r"""
def show_recommendations(title, similarity_matrix, n=3):
    recs = recommend(title, similarity_matrix, "multi-modal", n=n)
    selected_titles = [title] + recs["recommended_movie"].tolist()
    selected = movies[movies["title"].isin(selected_titles)].copy()
    selected["rank"] = selected["title"].map({movie_title: i for i, movie_title in enumerate(selected_titles)})
    selected = selected.sort_values("rank")
    show_poster_grid(selected, title=f"Query and Top-{n} recommendations for {title}")
    return recs

show_recommendations("Toy Story", multimodal_similarity, n=3)
"""),
        md("""
Interpretation:

Multi-modal systems combine evidence from more than one representation. Here students can inspect the poster images directly, then see how image features can be fused with text features. In a full environment the notebook can use CLIP embeddings; in a basic environment it still runs with simple poster pixel features.

Student task:
1. Change `image_weight` from `0.35` to `0.60`.
2. Try `show_recommendations("Heat", multimodal_similarity, n=3)`.
3. Which ranking changes are useful, and which look like visual noise?
"""),
    ]


def slide_exercise_05_context() -> list[dict]:
    return [
        md("""
# Slide Exercise 05: Context-Aware and Explainable Recommender

This is the refined version of `ContextAware_MovieRecommender.ipynb`.

Learning objectives:
- Build a base content recommendation score.
- Re-rank using context such as device, time, and family mode.
- Explain each recommendation with feature-level reasons.

Main functions used:
- `TfidfVectorizer(...)`: builds item content vectors.
- `cosine_similarity(...)`: computes base relevance.
- `DataFrame.loc[...]`: applies context bonuses to matching rows.
- Custom explanation functions: translate feature overlap into readable reasons.
"""),
        code(COMMON_LOAD),
        md("Start from a zero-shot style query so the context effect is easy to see."),
        code(r"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

movies["context_text"] = movies["genres"].str.replace("|", " ", regex=False) + " " + movies["description"] + " " + movies["keywords"]
vectorizer = TfidfVectorizer(stop_words="english")
item_matrix = vectorizer.fit_transform(movies["context_text"])

def base_recommend(query, n=8):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, item_matrix).ravel()
    results = movies[["title", "genres", "director", "duration_min", "family_friendly"]].copy()
    results["base_score"] = scores
    return results.sort_values("base_score", ascending=False).head(n)

base_recommend("family adventure comedy")
"""),
        md("Apply context rules as a transparent re-ranking layer."),
        code(r"""
def apply_context(results, context):
    reranked = results.copy()
    reranked["context_bonus"] = 0.0

    if context == "morning_mobile":
        reranked.loc[reranked["duration_min"] <= 110, "context_bonus"] += 0.12
    if context == "evening_tv":
        reranked.loc[reranked["duration_min"] >= 120, "context_bonus"] += 0.10
    if context == "family_mode":
        reranked.loc[reranked["family_friendly"] == 1, "context_bonus"] += 0.25

    reranked["final_score"] = reranked["base_score"] + reranked["context_bonus"]
    return reranked.sort_values("final_score", ascending=False)

base = base_recommend("family adventure comedy")
apply_context(base, "family_mode")
"""),
        md("Generate short explanations from shared query terms and metadata."),
        code(r"""
def explain(query, title):
    movie = movies[movies["title"].eq(title)].iloc[0]
    q = set(query.lower().split())
    genre_matches = [g for g in movie["genres"].split("|") if g.lower() in q]
    keyword_matches = [w for w in movie["keywords"].split() if w.lower() in q]
    reasons = genre_matches + keyword_matches
    if movie["family_friendly"] == 1:
        reasons.append("family-friendly")
    return "Recommended because it shares: " + ", ".join(reasons or ["related content"])

top = apply_context(base, "family_mode").iloc[0]["title"]
explain("family adventure comedy", top)
"""),
        md("""
Interpretation:

Context-aware recommendation does not replace the base recommender. It adjusts a reasonable ranking for the user's current situation.

Student task:
1. Add a `short_break` context that favors movies under 100 minutes.
2. Explain one recommendation before and after re-ranking.
"""),
    ]


def slide_exercise_06_zeroshot() -> list[dict]:
    return [
        md("""
# Slide Exercise 06: Zero-Shot and Generative Recommender

This is the refined version of `Zero_Shot_Generative_Recommender.ipynb` and matches the slide exercise name `ZeroShot_Generative_Recommender.ipynb`.

Learning objectives:
- Search movies with natural-language queries.
- Use TF-IDF as a reliable zero-shot baseline.
- Optionally upgrade to SBERT embeddings.
- Treat generative metadata enrichment as a reviewed stub, not a required API call.

Main functions used:
- `vectorizer.transform(...)`: converts a new query into the same feature space as movies.
- `cosine_similarity(...)`: ranks movies against the query.
- `try/except`: enables optional semantic embeddings safely.
- Custom stub functions: show where generative enrichment could be added.
"""),
        code(COMMON_LOAD),
        md("Zero-shot search means the user can write a request directly instead of choosing a known item."),
        code(r"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

movies["zero_shot_text"] = movies["title"] + " " + movies["genres"].str.replace("|", " ", regex=False) + " " + movies["description"] + " " + movies["keywords"]

vectorizer = TfidfVectorizer(stop_words="english")
item_matrix = vectorizer.fit_transform(movies["zero_shot_text"])

def zero_shot_search(query, n=5):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, item_matrix).ravel()
    results = movies[["title", "genres", "description"]].copy()
    results["score"] = scores
    return results.sort_values("score", ascending=False).head(n)

zero_shot_search("movies about space exploration")
"""),
        md("Try several natural-language requests."),
        code(r"""
queries = [
    "movies about space exploration",
    "light comedy for family evening",
    "romantic drama with music",
]

pd.concat(
    [zero_shot_search(q, n=3).assign(query=q) for q in queries],
    ignore_index=True,
)[["query", "title", "score", "genres"]]
"""),
        md("Optional semantic embeddings can improve zero-shot behavior, but the exercise remains complete without them."),
        code(r"""
semantic_available = False
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    semantic_item_matrix = model.encode(movies["zero_shot_text"].tolist(), show_progress_bar=False)
    semantic_available = True
except Exception as exc:
    print("Optional SBERT model is not available. Continue with TF-IDF zero-shot search.")
    print(type(exc).__name__, str(exc)[:160])

semantic_available
"""),
        md("Keep generative enrichment as a safe, reviewed placeholder."),
        code(r"""
def generative_metadata_enrichment_stub(title, short_description):
    return {
        "title": title,
        "suggested_tags": ["review-before-use", "course-demo", "generated-metadata-placeholder"],
        "draft_description": short_description,
        "warning": "Generated metadata should be reviewed before it changes recommendations.",
    }

generative_metadata_enrichment_stub(
    "Example New Movie",
    "A crew searches for a safe planet after Earth becomes difficult to inhabit.",
)
"""),
        md("""
Interpretation:

Zero-shot recommendation is useful for cold-start discovery and natural-language search. Generative metadata can help fill gaps, but it must be reviewed because generated content can be wrong or inconsistent.

Student task:
1. Write a query that should retrieve `The Matrix`.
2. Add a new movie with sparse metadata and test whether the query search can find it.
"""),
    ]


def write_slide_exercises() -> None:
    write_notebook("01_TFIDF_MovieRecommender_Expanded.ipynb", slide_exercise_01_tfidf(), EXERCISES)
    write_notebook("02_SBERT_MovieRecommender.ipynb", slide_exercise_02_sbert(), EXERCISES)
    write_notebook("03_GraphCB_MovieRecommender_NodeVectors.ipynb", slide_exercise_03_graph(), EXERCISES)
    write_notebook("04_MultiModal_MovieRecommender.ipynb", slide_exercise_04_multimodal(), EXERCISES)
    write_notebook("05_ContextAware_MovieRecommender.ipynb", slide_exercise_05_context(), EXERCISES)
    write_notebook("06_ZeroShot_Generative_Recommender.ipynb", slide_exercise_06_zeroshot(), EXERCISES)
    (EXERCISES / "README.md").write_text(
        """# Chapter 2 Slide Exercises

These notebooks are refined versions of the six exercise notebooks referenced in the Chapter 2 slides. They are kept separate from the seven core practical notebooks to avoid duplication:

| Slide exercise | Refined notebook | Source / note |
| --- | --- | --- |
| 1 | `01_TFIDF_MovieRecommender_Expanded.ipynb` | Refines `TFIDF_MovieRecommender_Expanded.ipynb` |
| 2 | `02_SBERT_MovieRecommender.ipynb` | Refines `SBERT_MovieRecommender.ipynb` with fallback logic |
| 3 | `03_GraphCB_MovieRecommender_NodeVectors.ipynb` | Refines graph recommendation without requiring node-vector packages |
| 4 | `04_MultiModal_MovieRecommender.ipynb` | Refines multi-modal fusion with real poster URLs and fallback image features |
| 5 | `05_ContextAware_MovieRecommender.ipynb` | Refines context-aware and explainable recommendation |
| 6 | `06_ZeroShot_Generative_Recommender.ipynb` | Slide-style name; source file in repo is `Zero_Shot_Generative_Recommender.ipynb` |

Each exercise includes a `Main functions used` section so students can connect the code to the library calls.
""",
        encoding="utf-8",
    )


def write_readme() -> None:
    (CHAPTER / "README.md").write_text(
        """# Chapter 2: Content-Based Recommendation

This folder contains the reorganized practical material for Chapter 2. The notebooks are ordered from simple feature vectors to modern content-based extensions, while staying runnable in a basic Python environment.

## Notebook Path

| Order | Notebook | Main idea |
| --- | --- | --- |
| 01 | `01_feature_vectors_and_similarity.ipynb` | Manual feature vectors, one-hot genres, cosine, Jaccard, Euclidean distance |
| 02 | `02_tfidf_movie_recommender.ipynb` | Bag-of-Words, TF-IDF, cosine similarity, Top-N similar movies |
| 03 | `03_user_profile_and_ranking.ipynb` | User profiles, weighted feedback, temporal decay, ranking explanations |
| 04 | `04_evaluation_topk_metrics.ipynb` | Precision@K, Recall@K, HitRate@K, metric table and chart |
| 05 | `05_sbert_semantic_recommender.ipynb` | SBERT semantic similarity with TF-IDF fallback |
| 06 | `06_graph_based_content_recommender.ipynb` | Content graph, NetworkX fallback, optional Node2Vec |
| 07 | `07_context_explainable_zero_shot_demo.ipynb` | Context re-ranking, explanations, zero-shot style search |

## Slide Exercises

The `slide_exercises/` folder contains refined versions of the six exercise notebooks referenced in the Chapter 2 slides:

1. `01_TFIDF_MovieRecommender_Expanded.ipynb`
2. `02_SBERT_MovieRecommender.ipynb`
3. `03_GraphCB_MovieRecommender_NodeVectors.ipynb`
4. `04_MultiModal_MovieRecommender.ipynb`
5. `05_ContextAware_MovieRecommender.ipynb`
6. `06_ZeroShot_Generative_Recommender.ipynb`

Each exercise includes a short `Main functions used` section. These notebooks are applied extensions, while the seven notebooks above remain the recommended first path for students.

## Data

The `data/` folder contains small local CSV files used across the notebooks:

- `movies_chapter2.csv`
- `user_interactions_chapter2.csv`

The examples deliberately use a small dataset so students can inspect intermediate tables and understand each step.

## HTML Demos

The `html_demos/` folder contains simple lecture demos:

- `index.html`
- `real_world_tfidf_search_demo.html`
- `semantic_vs_keyword_demo.html`
- `graph_content_explorer_demo.html`
- `multimodal_poster_recommender_demo.html`
- `context_aware_reranking_demo.html`
- `zero_shot_discovery_demo.html`
- `content_based_filtering_demo.html`
- `tfidf_similarity_demo.html`
- `user_profile_ranking_demo.html`
- `poi_content_based_demo.html`

These are standalone HTML files and can be opened directly in a browser. Start with `html_demos/index.html` for the richer real-world demos.

## Optional Dependencies

The core notebooks use `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `networkx`. SBERT and graph embedding examples include fallback logic, so missing optional packages should not break the practical path.
""",
        encoding="utf-8",
    )


def write_report() -> None:
    (CHAPTER / "CHAPTER_02_REORGANIZATION_REPORT.md").write_text(
        """# Chapter 2 Reorganization Report

## Repository Findings

The current repository stores Chapter 2 content-based notebooks at the repository root. The relevant existing files are:

- `TFIDF_MovieRecommender_Expanded.ipynb`
- `SBERT_MovieRecommender.ipynb`
- `GraphCB_MovieRecommender_NodeVectors.ipynb`
- `ContextAware_MovieRecommender.ipynb`
- `Zero_Shot_Generative_Recommender.ipynb`
- `MultiModal_MovieRecommender.ipynb`
- `POI_Recommender_Yelp.ipynb`

The repository also contains Chapter 3 collaborative filtering notebooks:

- `Chapter3_CF_Practical.ipynb`
- `Chapter3_CF_Practical_Enhanced.ipynb`

No existing HTML demo files were found during inspection.

## Issues Found

- Chapter 2 material was flat at the repository root rather than organized as a chapter folder.
- Several notebooks were advanced demos instead of a gradual beginner-friendly practical sequence.
- Some notebooks required optional packages or remote downloads at the top of the notebook.
- Evaluation, user profiles, and basic feature-vector similarity were not separated into clear dedicated practical notebooks.
- The existing root README described Chapter 2 as an advanced notebook collection, not as a structured practical path.

## Implemented Structure

Created:

```text
chapter_02_content_based/
  01_feature_vectors_and_similarity.ipynb
  02_tfidf_movie_recommender.ipynb
  03_user_profile_and_ranking.ipynb
  04_evaluation_topk_metrics.ipynb
  05_sbert_semantic_recommender.ipynb
  06_graph_based_content_recommender.ipynb
  07_context_explainable_zero_shot_demo.ipynb
  html_demos/
  data/
  slide_exercises/
  README.md
  CHAPTER_02_REORGANIZATION_REPORT.md
```

Added slide exercise notebooks:

```text
chapter_02_content_based/slide_exercises/
  01_TFIDF_MovieRecommender_Expanded.ipynb
  02_SBERT_MovieRecommender.ipynb
  03_GraphCB_MovieRecommender_NodeVectors.ipynb
  04_MultiModal_MovieRecommender.ipynb
  05_ContextAware_MovieRecommender.ipynb
  06_ZeroShot_Generative_Recommender.ipynb
  README.md
```

The six slide exercises were refined to remove duplicate setup patterns, avoid mandatory model downloads, and explain the main functions used in each exercise. The multi-modal exercise keeps the real poster URLs from the previous notebook so students can see visual recommendation results, while still including fallback image features if downloads or CLIP are unavailable. The slide name `ZeroShot_Generative_Recommender.ipynb` is preserved in the new exercise sequence; the existing source notebook in the repository is named `Zero_Shot_Generative_Recommender.ipynb`.

## Files to Keep, Move, or Deprecate Later

No files were deleted.

Recommended next cleanup, after review:

- Keep the old root-level Chapter 2 notebooks temporarily as archive/reference material.
- Consider moving them into an `archive/` or `legacy_chapter_02/` folder after the new notebooks are approved.
- Consider folding useful multi-modal content from `MultiModal_MovieRecommender.ipynb` into a future optional extension, because the requested seven-notebook structure does not include a full multi-modal practical.
- Keep `POI_Recommender_Yelp.ipynb` as a separate domain example or later adapt it into Chapter 2/Chapter 7 depending on the course structure.

## Slide Alignment

The new notebooks map to the Chapter 2 slide topics:

- Feature vectors and item representation: Notebook 01
- Bag-of-Words and TF-IDF: Notebook 02
- User profiles and ranking: Notebook 03
- Top-K evaluation: Notebook 04
- SBERT/BERT embeddings: Notebook 05
- Graph-based content recommendation: Notebook 06
- Context-aware, explainable, zero-shot, and generative stubs: Notebook 07

## Real-World HTML Demos

Additional standalone visual demos were added under `html_demos/` based on the slide exercises:

- `index.html`
- `real_world_tfidf_search_demo.html`
- `semantic_vs_keyword_demo.html`
- `graph_content_explorer_demo.html`
- `multimodal_poster_recommender_demo.html`
- `context_aware_reranking_demo.html`
- `zero_shot_discovery_demo.html`

These demos use real movie posters, interactive search, context controls, graph visualization, and visual recommendation cards so students can see recommendation behavior rather than only reading code output.
""",
        encoding="utf-8",
    )


def html_page(title: str, body: str, script: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f7f7f4;
      --ink: #1f2937;
      --line: #d8d8d0;
      --accent: #2563eb;
      --accent2: #0f766e;
      --card: #ffffff;
    }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    main {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 28px 18px 40px;
    }}
    h1 {{
      font-size: 28px;
      margin: 0 0 8px;
      letter-spacing: 0;
    }}
    p {{
      margin: 0 0 18px;
      line-height: 1.5;
    }}
    .panel {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin: 14px 0;
    }}
    .row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    button, select {{
      font: inherit;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      background: white;
    }}
    button {{
      cursor: pointer;
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px;
      text-align: left;
      vertical-align: top;
    }}
    .bar {{
      height: 12px;
      border-radius: 999px;
      background: #e5e7eb;
      overflow: hidden;
      min-width: 120px;
    }}
    .bar span {{
      display: block;
      height: 100%;
      background: var(--accent2);
    }}
    .tag {{
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      margin: 2px;
      background: #fafafa;
      font-size: 12px;
    }}
    svg {{
      max-width: 100%;
      height: auto;
      background: white;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
  </style>
</head>
<body>
<main>
{body}
</main>
<script>
const movies = {json.dumps(MOVIES)};
{script}
</script>
</body>
</html>
"""


def real_world_html_page(title: str, body: str, script: str) -> str:
    movie_data = [{**movie, "poster_url": POSTER_URLS.get(movie["title"], "")} for movie in MOVIES]
    page = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #f5f6f8;
      --ink: #172033;
      --muted: #667085;
      --panel: #ffffff;
      --line: #d6dbe4;
      --accent: #155eef;
      --accent2: #087443;
      --warn: #b54708;
      --shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      background: #101828;
      color: white;
      padding: 20px 22px;
      border-bottom: 4px solid var(--accent);
    }
    header h1 {
      margin: 0;
      font-size: clamp(24px, 3vw, 38px);
      letter-spacing: 0;
    }
    header p {
      color: #d0d5dd;
      max-width: 920px;
      margin: 8px 0 0;
      line-height: 1.5;
    }
    main { max-width: 1220px; margin: 0 auto; padding: 18px; }
    .toolbar, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 14px;
      margin: 14px 0;
    }
    .toolbar {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 12px;
      align-items: end;
    }
    label { display: block; font-size: 13px; color: var(--muted); margin-bottom: 5px; }
    input, select, button {
      width: 100%;
      font: inherit;
      border-radius: 6px;
      border: 1px solid var(--line);
      padding: 9px 10px;
      background: white;
      color: var(--ink);
    }
    button {
      cursor: pointer;
      background: var(--accent);
      color: white;
      border-color: var(--accent);
      font-weight: 700;
    }
    button.secondary { background: #ffffff; color: var(--accent); }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
      gap: 14px;
    }
    .card {
      background: white;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      min-height: 100%;
    }
    .card img {
      width: 100%;
      aspect-ratio: 2 / 3;
      object-fit: cover;
      display: block;
      background: #e5e7eb;
    }
    .card-body { padding: 10px; }
    .card h3 { margin: 0 0 5px; font-size: 15px; line-height: 1.25; }
    .meta { color: var(--muted); font-size: 12px; line-height: 1.35; }
    .tag {
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 7px;
      margin: 2px 2px 2px 0;
      font-size: 12px;
      background: #f8fafc;
    }
    .scorebar {
      height: 8px;
      border-radius: 999px;
      background: #e4e7ec;
      margin: 8px 0 4px;
      overflow: hidden;
    }
    .scorebar span { display: block; height: 100%; background: var(--accent2); }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { border-bottom: 1px solid var(--line); padding: 9px; text-align: left; vertical-align: top; }
    .split { display: grid; grid-template-columns: minmax(260px, 0.9fr) minmax(320px, 1.4fr); gap: 14px; }
    .mini { font-size: 12px; color: var(--muted); }
    svg { width: 100%; min-height: 460px; border: 1px solid var(--line); border-radius: 8px; background: white; }
    .pillrow { display: flex; gap: 8px; flex-wrap: wrap; }
    .pillrow button { width: auto; padding: 7px 10px; background: #eef4ff; color: #1849a9; border-color: #b2ccff; }
    @media (max-width: 760px) {
      .split { grid-template-columns: 1fr; }
      main { padding: 12px; }
    }
  </style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <p>__SUBTITLE__</p>
</header>
<main>
__BODY__
</main>
<script>
const movies = __MOVIES__;
const genreList = [...new Set(movies.flatMap(m => m.genres.split("|")))].sort();
function terms(text) {
  return String(text).toLowerCase().replace(/[^a-z0-9\\s-]/g, " ").split(/\\s+/).filter(w => w.length > 2);
}
function movieText(m) {
  return `${m.title} ${m.genres.replaceAll("|", " ")} ${m.director} ${m.description} ${m.keywords}`;
}
const docs = movies.map(movieText);
const vocab = [...new Set(docs.flatMap(terms))];
const idf = Object.fromEntries(vocab.map(t => [t, Math.log((1 + docs.length) / (1 + docs.filter(d => terms(d).includes(t)).length)) + 1]));
function vectorize(text) {
  const counts = {};
  terms(text).forEach(t => counts[t] = (counts[t] || 0) + 1);
  return Object.fromEntries(vocab.map(t => [t, (counts[t] || 0) * idf[t]]));
}
const docVectors = docs.map(vectorize);
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  vocab.forEach(t => { dot += (a[t] || 0) * (b[t] || 0); na += (a[t] || 0) ** 2; nb += (b[t] || 0) ** 2; });
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}
function genreSet(movie) { return new Set(movie.genres.split("|")); }
function overlap(a, b) { return [...a].filter(x => b.has(x)); }
function movieCard(item, score=0, note="") {
  const genres = item.genres.split("|").map(g => `<span class="tag">${g}</span>`).join("");
  return `<article class="card">
    <img src="${item.poster_url}" alt="${item.title} poster">
    <div class="card-body">
      <h3>${item.title}</h3>
      <div class="meta">${item.year} | ${item.director}</div>
      <div>${genres}</div>
      ${score ? `<div class="scorebar"><span style="width:${Math.max(3, Math.min(100, score * 100))}%"></span></div><div class="mini">score ${score.toFixed(3)}</div>` : ""}
      ${note ? `<p class="mini">${note}</p>` : ""}
    </div>
  </article>`;
}
function rankByQuery(query) {
  const qv = vectorize(query);
  return movies.map((m, i) => {
    const shared = vocab.filter(t => (qv[t] || 0) > 0 && (docVectors[i][t] || 0) > 0).slice(0, 8);
    return { ...m, score: cosine(qv, docVectors[i]), shared };
  }).sort((a, b) => b.score - a.score);
}
function semanticRank(query) {
  const synonymMap = {
    space: ["space", "astronaut", "mars", "orbit", "wormhole", "exploration"],
    family: ["family", "children", "animation", "kindness", "toys"],
    romance: ["romance", "love", "lifelong", "couple", "music"],
    mind: ["dreams", "simulation", "reality", "mind-bending", "hacker"],
    crime: ["crime", "justice", "chaos", "superhero", "action"],
    comedy: ["comedy", "humor", "light", "family"]
  };
  const expanded = terms(query).flatMap(t => synonymMap[t] || [t]).join(" ");
  return rankByQuery(query + " " + expanded);
}
__SCRIPT__
</script>
</body>
</html>
"""
    subtitle = body.split("\n", 1)[0].strip()
    clean_body = body.split("\n", 1)[1] if "\n" in body else body
    return (
        page.replace("__TITLE__", title)
        .replace("__SUBTITLE__", subtitle)
        .replace("__BODY__", clean_body)
        .replace("__MOVIES__", json.dumps(movie_data))
        .replace("__SCRIPT__", script)
    )


def write_real_world_html() -> None:
    (DEMOS / "index.html").write_text(
        real_world_html_page(
            "Chapter 2 Real-World Demo Hub",
            """Choose a demo below. These mini-apps connect the Chapter 2 slide exercises to visual, classroom-ready recommendation scenarios.
<section class="panel">
  <div class="grid" id="demoGrid"></div>
</section>""",
            """
const demos = [
  ["TF-IDF Search", "real_world_tfidf_search_demo.html", "Search a catalog and inspect shared weighted terms."],
  ["Semantic vs Keyword", "semantic_vs_keyword_demo.html", "Compare lexical and semantic-style recommendations."],
  ["Content Graph", "graph_content_explorer_demo.html", "Explore item nodes, edges, and graph-neighborhood ranking."],
  ["Multi-Modal Posters", "multimodal_poster_recommender_demo.html", "Fuse text and poster style with real movie artwork."],
  ["Context Re-Ranking", "context_aware_reranking_demo.html", "See how family, mobile, and TV contexts change ranking."],
  ["Zero-Shot Discovery", "zero_shot_discovery_demo.html", "Type natural-language intents and get recommendations."]
];
document.querySelector("#demoGrid").innerHTML = demos.map(([name, url, desc]) =>
  `<article class="card"><div class="card-body"><h3>${name}</h3><p class="meta">${desc}</p><p><a href="${url}">Open demo</a></p></div></article>`
).join("");
""",
        ),
        encoding="utf-8",
    )

    (DEMOS / "real_world_tfidf_search_demo.html").write_text(
        real_world_html_page(
            "Real-World TF-IDF Search Demo",
            """Search the catalog like a small streaming-service content search. The ranking shows scores and shared terms.
<section class="toolbar">
  <div><label>Search query</label><input id="query" value="space survival astronaut"></div>
  <div><label>Genre filter</label><select id="genre"><option value="">All genres</option></select></div>
  <div><button id="run">Run TF-IDF Search</button></div>
</section>
<section class="panel"><div id="results" class="grid"></div></section>
<section class="panel"><div id="table"></div></section>""",
            """
const genreSelect = document.querySelector("#genre");
genreList.forEach(g => genreSelect.add(new Option(g, g)));
function render() {
  const query = document.querySelector("#query").value;
  const genre = genreSelect.value;
  const ranked = rankByQuery(query).filter(m => !genre || m.genres.includes(genre)).slice(0, 6);
  document.querySelector("#results").innerHTML = ranked.map(m => movieCard(m, m.score, `Shared terms: ${m.shared.join(", ") || "weak lexical match"}`)).join("");
  document.querySelector("#table").innerHTML = `<table><thead><tr><th>Rank</th><th>Movie</th><th>Similarity</th><th>Shared TF-IDF terms</th></tr></thead><tbody>` +
    ranked.map((m, i) => `<tr><td>${i + 1}</td><td>${m.title}</td><td>${m.score.toFixed(3)}</td><td>${m.shared.join(", ") || "none"}</td></tr>`).join("") +
    `</tbody></table>`;
}
document.querySelector("#run").addEventListener("click", render);
render();
""",
        ),
        encoding="utf-8",
    )

    (DEMOS / "semantic_vs_keyword_demo.html").write_text(
        real_world_html_page(
            "Semantic vs Keyword Demo",
            """Compare exact-word TF-IDF ranking with a lightweight semantic expansion that simulates why embeddings can retrieve related meanings.
<section class="toolbar">
  <div><label>Intent</label><input id="query" value="mind bending reality"></div>
  <div><button id="run">Compare Methods</button></div>
</section>
<section class="split">
  <div class="panel"><h2>Keyword TF-IDF</h2><div id="keyword" class="grid"></div></div>
  <div class="panel"><h2>Semantic-style Expansion</h2><div id="semantic" class="grid"></div></div>
</section>""",
            """
function render() {
  const query = document.querySelector("#query").value;
  const keyword = rankByQuery(query).slice(0, 4);
  const semantic = semanticRank(query).slice(0, 4);
  document.querySelector("#keyword").innerHTML = keyword.map(m => movieCard(m, m.score, `Matched words: ${m.shared.join(", ") || "few exact terms"}`)).join("");
  document.querySelector("#semantic").innerHTML = semantic.map(m => movieCard(m, m.score, `Semantic-style match from expanded intent`)).join("");
}
document.querySelector("#run").addEventListener("click", render);
render();
""",
        ),
        encoding="utf-8",
    )

    (DEMOS / "graph_content_explorer_demo.html").write_text(
        real_world_html_page(
            "Graph-Based Content Explorer",
            """Click a seed movie and inspect graph edges based on shared genres and director. The table ranks graph-neighborhood recommendations.
<section class="toolbar">
  <div><label>Seed movie</label><select id="seed"></select></div>
  <div><button id="run">Explore Graph</button></div>
</section>
<section class="split">
  <div class="panel"><svg id="graph" viewBox="0 0 760 500"></svg></div>
  <div class="panel"><div id="ranking"></div></div>
</section>""",
            """
const seedSelect = document.querySelector("#seed");
movies.forEach(m => seedSelect.add(new Option(m.title, m.title)));
seedSelect.value = "Interstellar";
function edgeReason(a, b) {
  const shared = overlap(genreSet(a), genreSet(b));
  const sameDirector = a.director === b.director;
  const weight = shared.length + (sameDirector ? 1.5 : 0);
  return { weight, reason: [...shared, sameDirector ? "same director" : ""].filter(Boolean).join(", ") };
}
const positions = movies.map((m, i) => {
  const angle = (Math.PI * 2 * i) / movies.length - Math.PI / 2;
  return { title: m.title, x: 380 + 285 * Math.cos(angle), y: 250 + 185 * Math.sin(angle) };
});
function pos(title) { return positions.find(p => p.title === title); }
function render() {
  const seedTitle = seedSelect.value;
  const seed = movies.find(m => m.title === seedTitle);
  let edges = "";
  movies.forEach(a => movies.forEach(b => {
    if (a.title >= b.title) return;
    const e = edgeReason(a, b);
    if (e.weight > 0) {
      const pa = pos(a.title), pb = pos(b.title);
      const active = a.title === seedTitle || b.title === seedTitle;
      edges += `<line x1="${pa.x}" y1="${pa.y}" x2="${pb.x}" y2="${pb.y}" stroke="${active ? "#155eef" : "#d0d5dd"}" stroke-width="${active ? 3 : 1}" opacity="${active ? 0.9 : 0.45}"/>`;
    }
  }));
  const nodes = movies.map(m => {
    const p = pos(m.title);
    const isSeed = m.title === seedTitle;
    return `<g><circle cx="${p.x}" cy="${p.y}" r="${isSeed ? 23 : 17}" fill="${isSeed ? "#155eef" : "#eef4ff"}" stroke="#1849a9" stroke-width="2"></circle>
      <text x="${p.x}" y="${p.y + 38}" text-anchor="middle" font-size="11">${m.title}</text></g>`;
  }).join("");
  document.querySelector("#graph").innerHTML = edges + nodes;
  const rows = movies.filter(m => m.title !== seedTitle).map(m => {
    const direct = edgeReason(seed, m);
    const common = movies.filter(x => x.title !== seedTitle && x.title !== m.title && edgeReason(seed, x).weight > 0 && edgeReason(m, x).weight > 0).length;
    return { ...m, graphScore: direct.weight + 0.5 * common, reason: direct.reason || "shared graph neighborhood" };
  }).sort((a, b) => b.graphScore - a.graphScore).slice(0, 6);
  document.querySelector("#ranking").innerHTML = `<table><thead><tr><th>Rank</th><th>Movie</th><th>Graph score</th><th>Reason</th></tr></thead><tbody>` +
    rows.map((m, i) => `<tr><td>${i + 1}</td><td>${m.title}</td><td>${m.graphScore.toFixed(2)}</td><td>${m.reason}</td></tr>`).join("") +
    `</tbody></table>`;
}
document.querySelector("#run").addEventListener("click", render);
render();
""",
        ),
        encoding="utf-8",
    )

    (DEMOS / "multimodal_poster_recommender_demo.html").write_text(
        real_world_html_page(
            "Multi-Modal Poster Recommender",
            """Blend text similarity with poster-style similarity. Students can see how visual evidence changes the recommendation list.
<section class="toolbar">
  <div><label>Seed movie</label><select id="seed"></select></div>
  <div><label>Text weight</label><input id="textWeight" type="range" min="0" max="1" step="0.05" value="0.65"></div>
  <div><label>Poster weight</label><input id="imageWeight" type="range" min="0" max="1" step="0.05" value="0.35"></div>
  <div><button id="run">Recommend</button></div>
</section>
<section class="panel"><div id="weights" class="mini"></div><div id="results" class="grid"></div></section>""",
            """
const seedSelect = document.querySelector("#seed");
movies.forEach(m => seedSelect.add(new Option(m.title, m.title)));
seedSelect.value = "Toy Story";
const posterStyle = {
  "Inception": [0.18, 0.24, 0.34, 0.45], "Interstellar": [0.12, 0.18, 0.30, 0.52],
  "Titanic": [0.70, 0.60, 0.48, 0.82], "The Matrix": [0.10, 0.35, 0.25, 0.40],
  "Toy Story": [0.88, 0.70, 0.62, 0.92], "Finding Nemo": [0.80, 0.78, 0.72, 0.90],
  "The Dark Knight": [0.10, 0.13, 0.18, 0.35], "The Martian": [0.62, 0.46, 0.35, 0.66],
  "The Notebook": [0.72, 0.58, 0.56, 0.75], "Paddington": [0.90, 0.72, 0.54, 0.86],
  "Gravity": [0.12, 0.18, 0.22, 0.42], "La La Land": [0.82, 0.58, 0.68, 0.84]
};
function vecCos(a, b) {
  const dot = a.reduce((s, x, i) => s + x * b[i], 0);
  const na = Math.sqrt(a.reduce((s, x) => s + x * x, 0));
  const nb = Math.sqrt(b.reduce((s, x) => s + x * x, 0));
  return dot / (na * nb || 1);
}
function render() {
  const seed = movies.find(m => m.title === seedSelect.value);
  const seedIdx = movies.findIndex(m => m.title === seed.title);
  const textW = Number(document.querySelector("#textWeight").value);
  const imgW = Number(document.querySelector("#imageWeight").value);
  const rows = movies.filter(m => m.title !== seed.title).map(m => {
    const idx = movies.findIndex(x => x.title === m.title);
    const textScore = cosine(docVectors[seedIdx], docVectors[idx]);
    const imageScore = vecCos(posterStyle[seed.title], posterStyle[m.title]);
    return { ...m, score: textW * textScore + imgW * imageScore, textScore, imageScore };
  }).sort((a, b) => b.score - a.score).slice(0, 5);
  document.querySelector("#weights").textContent = `Text weight ${textW.toFixed(2)} | Poster weight ${imgW.toFixed(2)}`;
  document.querySelector("#results").innerHTML = [movieCard(seed, 0, "Query movie")].concat(
    rows.map(m => movieCard(m, m.score, `text ${m.textScore.toFixed(2)} | poster ${m.imageScore.toFixed(2)}`))
  ).join("");
}
document.querySelector("#run").addEventListener("click", render);
render();
""",
        ),
        encoding="utf-8",
    )

    (DEMOS / "context_aware_reranking_demo.html").write_text(
        real_world_html_page(
            "Context-Aware Re-Ranking Demo",
            """Start from a query, then apply morning mobile, evening TV, or family mode. The ranking changes for practical reasons.
<section class="toolbar">
  <div><label>Query</label><input id="query" value="family adventure comedy"></div>
  <div><label>Context</label><select id="context"><option value="none">No context</option><option value="morning_mobile">Morning mobile</option><option value="evening_tv">Evening TV</option><option value="family_mode">Family mode</option></select></div>
  <div><button id="run">Re-rank</button></div>
</section>
<section class="panel"><div id="results" class="grid"></div></section>""",
            """
function contextBonus(m, context) {
  if (context === "morning_mobile" && Number(m.duration_min) <= 110) return [0.12, "short enough for mobile"];
  if (context === "evening_tv" && Number(m.duration_min) >= 120) return [0.10, "longer evening viewing"];
  if (context === "family_mode" && Number(m.family_friendly) === 1) return [0.22, "family-friendly"];
  return [0, "base content score"];
}
function render() {
  const query = document.querySelector("#query").value;
  const context = document.querySelector("#context").value;
  const ranked = rankByQuery(query).map(m => {
    const [bonus, reason] = contextBonus(m, context);
    return { ...m, finalScore: m.score + bonus, reason, bonus };
  }).sort((a, b) => b.finalScore - a.finalScore).slice(0, 6);
  document.querySelector("#results").innerHTML = ranked.map(m => movieCard(m, m.finalScore, `${m.reason}; context bonus ${m.bonus.toFixed(2)}`)).join("");
}
document.querySelector("#run").addEventListener("click", render);
render();
""",
        ),
        encoding="utf-8",
    )

    (DEMOS / "zero_shot_discovery_demo.html").write_text(
        real_world_html_page(
            "Zero-Shot Discovery Demo",
            """Type what you want in natural language. The app uses query expansion to simulate zero-shot semantic discovery without an API key.
<section class="toolbar">
  <div><label>Natural-language request</label><input id="query" value="movies about space exploration and survival"></div>
  <div><button id="run">Discover Movies</button></div>
</section>
<section class="panel"><div class="pillrow" id="examples"></div></section>
<section class="panel"><div id="results" class="grid"></div></section>""",
            """
const examples = [
  "movies about space exploration and survival",
  "light comedy for family evening",
  "romantic drama with music",
  "mind bending reality simulation",
  "dark action crime justice"
];
document.querySelector("#examples").innerHTML = examples.map(q => `<button type="button" data-q="${q}">${q}</button>`).join("");
document.querySelector("#examples").addEventListener("click", event => {
  if (event.target.dataset.q) {
    document.querySelector("#query").value = event.target.dataset.q;
    render();
  }
});
function render() {
  const query = document.querySelector("#query").value;
  const ranked = semanticRank(query).slice(0, 6);
  document.querySelector("#results").innerHTML = ranked.map(m => movieCard(m, m.score, `Zero-shot intent match: ${m.genres.replaceAll("|", ", ")}`)).join("");
}
document.querySelector("#run").addEventListener("click", render);
render();
""",
        ),
        encoding="utf-8",
    )


def write_html() -> None:
    (DEMOS / "content_based_filtering_demo.html").write_text(
        html_page(
            "Content-Based Filtering Demo",
            """<h1>Content-Based Filtering Demo</h1>
<p>Select a movie and inspect how shared genres and director overlap affect similarity.</p>
<section class="panel">
  <label for="movieSelect">Seed movie</label>
  <select id="movieSelect"></select>
  <button id="runButton">Rank Similar Movies</button>
</section>
<section class="panel"><div id="features"></div></section>
<section class="panel"><div id="results"></div></section>""",
            r"""
const select = document.querySelector("#movieSelect");
movies.forEach((m, i) => select.add(new Option(m.title, i)));

function genreSet(movie) { return new Set(movie.genres.split("|")); }
function similarity(a, b) {
  const ga = genreSet(a), gb = genreSet(b);
  const intersection = [...ga].filter(g => gb.has(g));
  const union = new Set([...ga, ...gb]);
  const genreScore = intersection.length / union.size;
  const directorBonus = a.director === b.director ? 0.25 : 0;
  return { score: genreScore + directorBonus, shared: intersection, director: directorBonus > 0 };
}
function render() {
  const seed = movies[Number(select.value)];
  document.querySelector("#features").innerHTML = `<strong>${seed.title}</strong><br>` +
    seed.genres.split("|").map(g => `<span class="tag">${g}</span>`).join("") +
    `<span class="tag">${seed.director}</span>`;
  const rows = movies.filter(m => m.title !== seed.title).map(m => ({ movie: m, ...similarity(seed, m) }))
    .sort((a, b) => b.score - a.score);
  document.querySelector("#results").innerHTML = `<table><thead><tr><th>Rank</th><th>Movie</th><th>Shared content</th><th>Score</th></tr></thead><tbody>` +
    rows.map((r, i) => `<tr><td>${i + 1}</td><td>${r.movie.title}</td><td>${r.shared.join(", ") || "No shared genres"}${r.director ? ", same director" : ""}</td><td><div class="bar"><span style="width:${Math.min(100, r.score * 100)}%"></span></div>${r.score.toFixed(2)}</td></tr>`).join("") +
    `</tbody></table>`;
}
document.querySelector("#runButton").addEventListener("click", render);
render();
""",
        ),
        encoding="utf-8",
    )

    (DEMOS / "tfidf_similarity_demo.html").write_text(
        html_page(
            "TF-IDF Similarity Demo",
            """<h1>TF-IDF Similarity Demo</h1>
<p>Type a short query and see which movies share the most distinctive terms.</p>
<section class="panel">
  <input id="query" value="space survival astronaut" style="width:min(520px, 90%); padding:8px; border:1px solid var(--line); border-radius:6px;">
  <button id="searchButton">Search</button>
</section>
<section class="panel"><div id="results"></div></section>""",
            r"""
function words(text) {
  return text.toLowerCase().replace(/[^a-z0-9\s-]/g, " ").split(/\s+/).filter(w => w.length > 2);
}
const docs = movies.map(m => `${m.title} ${m.genres.replaceAll("|", " ")} ${m.director} ${m.description} ${m.keywords}`);
const vocab = [...new Set(docs.flatMap(words))];
const idf = Object.fromEntries(vocab.map(term => [term, Math.log((1 + docs.length) / (1 + docs.filter(d => words(d).includes(term)).length)) + 1]));
function vector(text) {
  const counts = {};
  words(text).forEach(w => counts[w] = (counts[w] || 0) + 1);
  return Object.fromEntries(vocab.map(t => [t, (counts[t] || 0) * idf[t]]));
}
const docVectors = docs.map(vector);
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  vocab.forEach(t => { dot += a[t] * b[t]; na += a[t] ** 2; nb += b[t] ** 2; });
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}
function render() {
  const q = document.querySelector("#query").value;
  const qv = vector(q);
  const rows = movies.map((m, i) => {
    const shared = vocab.filter(t => qv[t] > 0 && docVectors[i][t] > 0).slice(0, 8);
    return { title: m.title, score: cosine(qv, docVectors[i]), shared };
  }).sort((a, b) => b.score - a.score).slice(0, 6);
  document.querySelector("#results").innerHTML = `<table><thead><tr><th>Movie</th><th>Shared terms</th><th>Similarity</th></tr></thead><tbody>` +
    rows.map(r => `<tr><td>${r.title}</td><td>${r.shared.join(", ") || "Related by weak terms"}</td><td><div class="bar"><span style="width:${Math.min(100, r.score * 100)}%"></span></div>${r.score.toFixed(3)}</td></tr>`).join("") +
    `</tbody></table>`;
}
document.querySelector("#searchButton").addEventListener("click", render);
render();
""",
        ),
        encoding="utf-8",
    )

    (DEMOS / "user_profile_ranking_demo.html").write_text(
        html_page(
            "User Profile Ranking Demo",
            """<h1>User Profile Ranking Demo</h1>
<p>Select liked movies. The demo builds a genre profile and ranks unseen movies.</p>
<section class="panel"><div id="choices" class="row"></div><button id="rankButton">Build Profile and Rank</button></section>
<section class="panel"><div id="profile"></div></section>
<section class="panel"><div id="results"></div></section>""",
            r"""
const choices = document.querySelector("#choices");
movies.forEach((m, i) => {
  choices.insertAdjacentHTML("beforeend", `<label><input type="checkbox" value="${i}" ${["Inception","Interstellar"].includes(m.title) ? "checked" : ""}> ${m.title}</label>`);
});
function render() {
  const selected = [...document.querySelectorAll("input:checked")].map(x => movies[Number(x.value)]);
  const profile = {};
  selected.forEach(m => m.genres.split("|").forEach(g => profile[g] = (profile[g] || 0) + 1));
  Object.keys(profile).forEach(g => profile[g] = profile[g] / selected.length);
  document.querySelector("#profile").innerHTML = `<strong>Profile:</strong> ` + Object.entries(profile).sort((a,b)=>b[1]-a[1]).map(([g,v]) => `<span class="tag">${g}: ${v.toFixed(2)}</span>`).join("");
  const selectedTitles = new Set(selected.map(m => m.title));
  const rows = movies.filter(m => !selectedTitles.has(m.title)).map(m => {
    const score = m.genres.split("|").reduce((s, g) => s + (profile[g] || 0), 0);
    const why = m.genres.split("|").filter(g => profile[g]);
    return { title: m.title, score, why };
  }).sort((a,b)=>b.score-a.score).slice(0,6);
  document.querySelector("#results").innerHTML = `<table><thead><tr><th>Rank</th><th>Movie</th><th>Why</th><th>Score</th></tr></thead><tbody>` +
    rows.map((r,i) => `<tr><td>${i+1}</td><td>${r.title}</td><td>${r.why.join(", ") || "Exploration candidate"}</td><td><div class="bar"><span style="width:${Math.min(100, r.score * 50)}%"></span></div>${r.score.toFixed(2)}</td></tr>`).join("") +
    `</tbody></table>`;
}
document.querySelector("#rankButton").addEventListener("click", render);
render();
""",
        ),
        encoding="utf-8",
    )

    (DEMOS / "poi_content_based_demo.html").write_text(
        html_page(
            "POI Content-Based Demo",
            """<h1>POI Content-Based Demo</h1>
<p>A small place-recommendation example using category, price, and distance preferences.</p>
<section class="panel">
  <label>Preferred category <select id="category"><option>Cafe</option><option>Museum</option><option>Park</option><option>Restaurant</option></select></label>
  <label>Max price <select id="price"><option value="1">1</option><option value="2" selected>2</option><option value="3">3</option></select></label>
  <button id="poiButton">Rank Places</button>
</section>
<section class="panel"><div id="results"></div></section>""",
            r"""
const places = [
  {name:"Riverside Cafe", category:"Cafe", tags:["quiet","coffee","wifi"], price:2, distance:0.8},
  {name:"Modern Art Museum", category:"Museum", tags:["art","indoor","culture"], price:3, distance:2.1},
  {name:"City Garden", category:"Park", tags:["family","walk","outdoor"], price:1, distance:1.2},
  {name:"Pasta House", category:"Restaurant", tags:["dinner","family","italian"], price:2, distance:1.7},
  {name:"Student Espresso", category:"Cafe", tags:["budget","coffee","quick"], price:1, distance:0.4}
];
function render() {
  const cat = document.querySelector("#category").value;
  const maxPrice = Number(document.querySelector("#price").value);
  const rows = places.map(p => {
    const categoryScore = p.category === cat ? 1 : 0;
    const priceScore = p.price <= maxPrice ? 0.5 : -0.3;
    const distanceScore = Math.max(0, 1 - p.distance / 3);
    return {...p, score: categoryScore + priceScore + distanceScore};
  }).sort((a,b)=>b.score-a.score);
  document.querySelector("#results").innerHTML = `<table><thead><tr><th>Place</th><th>Features</th><th>Score</th></tr></thead><tbody>` +
    rows.map(r => `<tr><td>${r.name}</td><td><span class="tag">${r.category}</span><span class="tag">price ${r.price}</span><span class="tag">${r.distance} km</span>${r.tags.map(t=>`<span class="tag">${t}</span>`).join("")}</td><td><div class="bar"><span style="width:${Math.min(100, r.score * 35)}%"></span></div>${r.score.toFixed(2)}</td></tr>`).join("") +
    `</tbody></table>`;
}
document.querySelector("#poiButton").addEventListener("click", render);
render();
""",
        ),
        encoding="utf-8",
    )


def update_root_readme() -> None:
    path = ROOT / "README.md"
    current = path.read_text(encoding="utf-8")
    marker = "## Recommended Chapter 2 Practical Path"
    setup_line = "!pip install sentence-transformers torch torchvision transformers networkx scikit-learn pillow tqdm"
    if setup_line in current and setup_line + "\n```" not in current:
        current = current.replace(setup_line, setup_line + "\n```", 1)
    block = """## Recommended Chapter 2 Practical Path

The reorganized Chapter 2 practical material now lives in [`chapter_02_content_based/`](./chapter_02_content_based/).

Start there for the student-facing sequence:

1. Feature vectors and similarity
2. TF-IDF movie recommendation
3. User profiles and ranking
4. Top-K evaluation metrics
5. SBERT semantic recommendation with fallback logic
6. Graph-based content recommendation
7. Context-aware, explainable, and zero-shot demos

The older root-level Chapter 2 notebooks are kept for reference until the new structure is reviewed.

"""
    if marker not in current:
        path.write_text(current.rstrip() + "\n\n---\n\n" + block, encoding="utf-8")


def main() -> None:
    write_data()
    write_notebook("01_feature_vectors_and_similarity.ipynb", notebook_01())
    write_notebook("02_tfidf_movie_recommender.ipynb", notebook_02())
    write_notebook("03_user_profile_and_ranking.ipynb", notebook_03())
    write_notebook("04_evaluation_topk_metrics.ipynb", notebook_04())
    write_notebook("05_sbert_semantic_recommender.ipynb", notebook_05())
    write_notebook("06_graph_based_content_recommender.ipynb", notebook_06())
    write_notebook("07_context_explainable_zero_shot_demo.ipynb", notebook_07())
    write_slide_exercises()
    write_readme()
    write_report()
    write_html()
    write_real_world_html()
    update_root_readme()


if __name__ == "__main__":
    main()
