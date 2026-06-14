# Chapter 3: Collaborative Filtering - Memory-Based

This folder contains the reorganized practical material for Chapter 3. The notebooks move from the user-item matrix to user-user, item-item, evaluation, mitigation, and graph-based collaborative filtering.

## Notebook Path

| Order | Notebook | Main idea |
| --- | --- | --- |
| 01 | `01_user_item_matrix_and_sparsity.ipynb` | User-item matrix, missing values, density, sparsity, co-rating overlap |
| 02 | `02_similarity_measures_for_cf.ipynb` | Cosine, Pearson, Jaccard, minimum overlap, shrinkage |
| 03 | `03_user_user_knn_prediction.ipynb` | User-user nearest neighbors, mean-centered weighted prediction, explanations |
| 04 | `04_item_item_collaborative_filtering.ipynb` | Item-item similarity, item-based prediction, cached similarity intuition |
| 05 | `05_evaluation_precision_recall_mae.ipynb` | Leave-one-out split, MAE, Precision@K, Recall@K, HitRate@K |
| 06 | `06_cold_start_sparsity_clustering_temporal.ipynb` | Cold start, sparsity, shrinkage, clustering, time decay |
| 07 | `07_graph_ppr_explainable_hybrid_cf.ipynb` | Bipartite graph, Personalized PageRank, path explanations, hybrid fallback |

These seven notebooks are the single recommended student path for Chapter 3. Extra exercise copies are not kept in a separate folder, so students do not have to choose between duplicate paths.

## Data

The `data/` folder contains small local CSV files used across the notebooks:

- `movies_chapter3.csv`
- `ratings_chapter3.csv`

The examples use a compact movie-rating matrix based on the Chapter 3 lecture examples, with a few extra users and movies for evaluation, clustering, temporal dynamics, and graph demos.

## HTML Demos

The `html_demos/` folder contains one standalone real-world classroom demo:

- `index.html`

This single page uses real movie posters and combines the main Chapter 3 ideas in one cinema recommendation application: the user-item matrix, sparsity, user-user similarity, item-item similarity, kNN prediction, cold-start fallback, temporal weighting, graph-style explanation, and Top-K evaluation. It can be opened directly in a browser.

## Optional Dependencies

The core notebooks use `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `networkx`. The examples are deliberately small so students can inspect intermediate tables and understand each step.
