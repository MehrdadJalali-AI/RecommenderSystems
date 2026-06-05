# Chapter 2 Reorganization Report

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
