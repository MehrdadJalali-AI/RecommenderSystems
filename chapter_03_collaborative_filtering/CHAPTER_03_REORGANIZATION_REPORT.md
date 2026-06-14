# Chapter 3 Reorganization Report

## Source Alignment

The structure is based on `RS_C3_V2.pdf`, Chapter 3: Collaborative Filtering - Memory-Based. The deck topics include:

- Why collaborative filtering and when it works.
- Memory-based collaborative filtering.
- User-user and item-item CF.
- User-item rating matrices and sparsity.
- Cosine, Pearson, and Jaccard similarity.
- Mean-centered kNN prediction.
- Evaluation with Precision@K, Recall@K, and MAE.
- Cold start, sparsity, popularity bias, scalability, and mitigations.
- Clustering, temporal dynamics, graph-based CF, Personalized PageRank, explainability, hybrid fallback, and POI extensions.

## Repository Findings

The repository already contained two root-level Chapter 3 notebooks:

- `Chapter3_CF_Practical.ipynb`
- `Chapter3_CF_Practical_Enhanced.ipynb`

Those notebooks remain at the repository root as reference material. They are not copied into the Chapter 3 folder, because the chapter folder should expose one clear student path.

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
  07_graph_ppr_explainable_hybrid_cf.ipynb
  data/
  html_demos/
  README.md
  CHAPTER_03_REORGANIZATION_REPORT.md
```

## Design Choices

- Kept the practical path beginner-friendly and sequential, mirroring the Chapter 2 organization.
- Used local CSV files so notebooks run without remote downloads.
- Used a compact rating matrix close to the lecture examples, with enough additional interactions for evaluation and graph examples.
- Added minimum-overlap, shrinkage, clustering, temporal weighting, graph PPR, and hybrid fallback examples because these are explicitly covered in the Chapter 3 slides.
- Left the existing root-level notebooks untouched while keeping the Chapter 3 folder focused on the seven practical notebooks.

## Real-World HTML Demo

The chapter keeps one standalone visual demo under `html_demos/`:

- `index.html`

The single page combines the Chapter 3 practical topics into one collaborative-filtering cinema recommender with real posters, rating-matrix inspection, user-user and item-item prediction, cold-start fallback, temporal weighting, graph-style explanation, and Top-K evaluation.

## Suggested Cleanup Later

- After review, consider archiving or moving the old root-level Chapter 3 notebooks.
- If the course uses MovieLens in class, optionally add one advanced notebook that repeats the same pipeline on `Datasets/ml-latest-small`.
- The POI topic is currently represented in the root `POI_Recommender_Yelp.ipynb`; it can later become an optional Chapter 3 extension or a later applied module.
