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

These seven notebooks are the single recommended student path for Chapter 2. Extra slide-exercise copies are not kept in a separate folder, so students do not have to choose between duplicate paths.

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
