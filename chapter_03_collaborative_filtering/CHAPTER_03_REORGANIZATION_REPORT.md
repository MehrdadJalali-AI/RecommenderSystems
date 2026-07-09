# Chapter 3 Reorganization Report

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
