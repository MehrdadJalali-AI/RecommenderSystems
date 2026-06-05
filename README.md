# 🎓 Advanced Recommender Systems – Course Materials  
This repository contains practical Jupyter/Colab notebooks developed for **Chapter 2: Advanced Recommender Systems** of the course *Applied Recommender Systems*.  
Each notebook demonstrates a specific recommendation paradigm — from classical TF-IDF models to multimodal, context-aware, and zero-shot generative approaches.

---

## 📘 Chapter 2 – Advanced Recommender Systems

| Notebook | Description | Core Techniques |
|-----------|--------------|----------------|
| **[`TFIDF_MovieRecommender_Expanded.ipynb`](./TFIDF_MovieRecommender_Expanded.ipynb)** | A **content-based baseline recommender** using TF-IDF on movie plots to compute cosine similarity between items. | `TF-IDF`, `cosine similarity`, `text preprocessing`, `content-based filtering` |
| **[`SBERT_MovieRecommender.ipynb`](./SBERT_MovieRecommender.ipynb)** | An upgraded semantic recommender using **Sentence-BERT** embeddings for deeper contextual understanding of movie plots. | `Sentence-Transformers`, `semantic embeddings`, `transformer models` |
| **[`GraphCB_MovieRecommender_NodeVectors.ipynb`](./GraphCB_MovieRecommender_NodeVectors.ipynb)** | A **graph-based content recommender** representing movies as nodes and relationships (genre/director) as edges. | `NetworkX`, `Node2Vec`, `GraphSAGE`, `embedding propagation`, `graph visualization` |
| **[`MultiModal_MovieRecommender.ipynb`](./MultiModal_MovieRecommender.ipynb)** | A **multi-modal recommender** that fuses text and image embeddings (e.g., plots + posters) to enhance similarity scoring. | `CLIP`, `ResNet`, `Sentence-BERT`, `feature fusion`, `cosine similarity` |
| **[`ContextAware_MovieRecommender.ipynb`](./ContextAware_MovieRecommender.ipynb)** | A **context-aware and explainable recommender** that re-ranks movies using contextual factors (e.g., time of day or mood). | `context re-ranking`, `explainability`, `feature attribution`, `visual interpretation` |
| **[`Zero_Shot_Generative_Recommender.ipynb`](./Zero_Shot_Generative_Recommender.ipynb)** | A **zero-shot / generative embedding system** that uses large language and vision models to recommend unseen items. | `LLMs`, `Sentence-Transformers`, `OpenAI API`, `semantic search`, `generative text embeddings` |

---

## 🧩 Learning Objectives
Through these notebooks, learners will:
- Understand and compare **different recommendation paradigms** (textual, visual, graph, contextual, generative).  
- Apply **modern embedding techniques** from TF-IDF → BERT → CLIP → LLMs.  
- Explore **multi-modal fusion** and **graph learning** strategies.  
- Implement **context-aware personalization** and **explainability** methods.  
- Experiment with **zero-shot reasoning** and **generative AI** in recommendation systems.

---

## ⚙️ Environment Setup
All notebooks are **Google Colab–ready** and require the following Python libraries:

```bash
!pip install sentence-transformers torch torchvision transformers networkx scikit-learn pillow tqdm
```

---

## Recommended Chapter 2 Practical Path

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
