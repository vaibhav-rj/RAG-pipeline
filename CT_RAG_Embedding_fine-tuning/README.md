# Clinical-Trials Embedding Fine-Tuning

## Overview

Domain-specific fine-tuning of `nomic-embed-text-v1.5` for **Clinical-Trials (CT) retrieval**.

The objective is to improve semantic representation and **positive-vs-negative retrieval discrimination within the Clinical-Trials domain**, rather than optimize cross-domain generalization.

## Experimental Pipeline

```text
Clinical-Trial Data
      ↓
Cleaning & Eligibility Processing
      ↓
Chunk Construction
      ↓
Synthetic Anchor Generation
      ↓
Anchor–Positive Dataset
      ↓
Nomic Embedding Fine-Tuning
(MNRL + Matryoshka)
      ↓
768 / 512 / 256 / 128 / 64D
      ↓
IR Evaluation & Model Comparison
```

## Dataset Formulations

Two alternative formulations were investigated:

### Old / Coarser formulation

**4 synthetic anchors → 1 consolidated positive**

`Title + Summary + Inclusion Criteria`

Two preprocessing/training iterations were explored, with progressively improved inclusion/exclusion parsing and chunk-length normalization.

### New / Granular formulation

**5 synthetic anchors → 3 positive chunks**

`Title → 1 | Summary → 2 | Inclusion Criteria → 2`

Two iterations were investigated with different relevant-document identification strategies. The second orientation produced below-par results and was not retained as a final model.

## Fine-Tuning

**Base model:** `nomic-ai/nomic-embed-text-v1.5`

**Training objectives:**

* Multiple Negatives Ranking Loss (MNRL)
* Matryoshka Representation Learning

**Embedding dimensions evaluated:**

`768 | 512 | 256 | 128 | 64`

## Evaluation

Retrieval performance was evaluated using:

* Accuracy@K
* Precision@K
* Recall@K
* NDCG@10
* MRR@10
* MAP@100

Both aggregate IR metrics and sample-level comparisons against other embedding models were examined.

## Key Outcome

The final CT-adapted embedding demonstrated improved **domain-specific retrieval behaviour** over the corresponding pretrained baseline, while retaining useful representations at reduced embedding dimensions.

The experiments also showed that **dataset formulation and anchor/chunk design materially affected retrieval performance**, leading to iterative refinement of the training data rather than simply increasing model complexity.

## Artifacts

Corresponding datasets and model iterations are published on Hugging Face:

**Datasets**

* [Iteration 1](https://huggingface.co/datasets/vab46/Clinical_trials_anchor-positive-pairs_EmbeddingModel-data)
* [Granular formulation](https://huggingface.co/datasets/vab46/Clinical_trials_anchor-positive-pairs_EmbeddingModel-data2)
* [Final dataset](https://huggingface.co/datasets/vab46/Clinical_trials_anchor-positive-pairs_EmbeddingModel-data_final)

**Models**

* [Iteration 1](https://huggingface.co/vab46/nomic-embed-text-v1.5_Clinical-Trials_Matryoshka)
* [Granular formulation](https://huggingface.co/vab46/nomic-embed-text-v1.5_Clinical-Trials_Matryoshka2)
* [Final model](https://huggingface.co/vab46/nomic-embed-text-v1.5_Clinical-Trials_Matryoshka_final)

Detailed preprocessing, modelling and iteration-level evaluation are available in the accompanying notebooks and result files.

## Next Stage

This embedding adaptation forms the retrieval foundation for subsequent:

**Query Encoder Fine-Tuning → LLM LoRA/PEFT Fine-Tuning → End-to-End CT RAG Optimization**
