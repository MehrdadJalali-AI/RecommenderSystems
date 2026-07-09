# Chapter 3: Collaborative Filtering - Memory-Based

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
