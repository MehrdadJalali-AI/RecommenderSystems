# Chapter 2: Content-Based Recommendation

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

- `content_based_filtering_demo.html`
- `tfidf_similarity_demo.html`
- `user_profile_ranking_demo.html`
- `poi_content_based_demo.html`

These are standalone HTML files and can be opened directly in a browser.

## Optional Dependencies

The core notebooks use `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `networkx`. SBERT and graph embedding examples include fallback logic, so missing optional packages should not break the practical path.
