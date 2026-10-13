# RAG1 — Clinical-Trial Embedding Fine-Tuning

## Objective

RAG1 specializes the **document embedding model** for clinical-trial retrieval.

> **Generic semantic similarity → clinical-trial-specific retrieval**

The goal was to improve retrieval of clinically relevant information, particularly for eligibility, age limits, conditions, interventions, thresholds, and trial-specific criteria.

---

## Pipeline

```mermaid
flowchart LR
    A[Clinical Trial JSON] --> B[Preprocessing]
    B --> C[Title + Summary + Inclusion Criteria]
    C --> D[Anchor / Question Generation]
    D --> E[90:10 Anchor Split]
    E --> F[Matryoshka MNRL Training]
    F --> G[Fine-tuned Nomic 768D]
    G --> H[Normalized Embeddings]
    H --> I[FAISS IndexFlatIP]
    I --> J[Top-k Retrieval]
```

---

## Data Preparation

- Source: ~10K clinical-trial records.
- Trial-level information was transformed into retrieval records.
- Final retrieval representation:

```text
Title + Summary + Inclusion Criteria
```

- Duplicate/near-duplicate trials were handled using cosine similarity.
- KNN clustering reduced the dataset to approximately **2K representative records**.

### Why consolidated chunks?

Experiments showed that aggressively separating trial information fragmented semantics.

The final design preserved the important trial-level relationship between:

```text
Trial identity
     +
Study description
     +
Eligibility information
```

---

## Anchor Generation

### Iteration 1

>Across all 3 iterations synthetic anchors were generated using **llama-3.1-8b instruct** model ran on derive chunks.

Four question types were generated from a trial/chunk:

1. Macro question
2. Patient-profile question
3. Operational question
4. Conversational question

### Iteration 2

A more granular strategy generated:

- Title → 1 anchor
- Summary → 2 anchors
- Inclusion criteria → 2 anchors

This increased query specificity but **fragmented trial-level semantics**, causing retrieval performance to deteriorate.

### Iteration 3

Returned to:

```text
Title + Summary + Inclusion Criteria
```

while retaining improved eligibility extraction/cleaning.

This recovered retrieval performance and became the final RAG1 representation.

---

## Train/Test & Evaluation Design

The split was performed at the **anchor/question level (90:10)**.

This evaluates:

> **New clinical-trial questions over an established clinical-trial knowledge base.**

A stricter trial-level holdout would instead test generalization to completely unseen trials.

The searchable evaluation corpus contains the full corpus, while the query set contains test anchors.

This is intentional because deployment also involves answering new questions against an existing knowledge base.

### InformationRetrievalEvaluator

The evaluator used:

```text
Corpus:
{document_id → document/chunk}

Queries:
{query_id → anchor/question}

Relevant documents:
{query_id → relevant document/chunk ID}
```

Sibling anchors generated from the same chunk were linked to that chunk as relevant documents.

---

## Training

### Model

Final fine-tuned model:

```text
vab46/nomic-embed-text-v1.5_Clinical-Trials_Matryoshka_final
```

### Objective

**Matryoshka Multiple Negatives Ranking Loss**

The model was trained simultaneously for:

```text
768D
512D
256D
128D
64D
```

Matryoshka training encourages useful representations across multiple embedding dimensions.

Final production retrieval uses **768D**.

---

## Iteration Evolution

| Iteration | Main change | Observation |
|---|---|---|
| Iter-1 | Coarse consolidated chunks + basic regex | Lower retrieval performance |
| Iter-2 | More granular title/summary/inclusion anchors | Retrieval deteriorated |
| Iter-3 | Consolidated chunks + improved eligibility cleaning | Retrieval substantially recovered |

### Key lesson

> More granular information does not automatically produce better retrieval. For clinical trials, preserving trial-level semantic relationships was more useful than aggressively fragmenting the context.

---

## Final Iteration-3 Retrieval Results

| Dimension | R@1 | R@3 | R@5 | R@10 | NDCG@10 | MRR@10 |
|---:|---:|---:|---:|---:|---:|---:|
| 768 | 0.5476 | 0.6620 | 0.6938 | 0.7395 | **0.6433** | **0.6126** |
| 512 | 0.5349 | 0.6417 | 0.6874 | 0.7395 | 0.6346 | 0.6015 |
| 256 | 0.5235 | 0.6531 | 0.6836 | 0.7306 | 0.6280 | 0.5951 |
| 128 | 0.4905 | 0.6048 | 0.6607 | 0.7116 | 0.5974 | 0.5612 |
| 64 | 0.4409 | 0.5667 | 0.6213 | 0.6773 | 0.5563 | 0.5179 |

Additional 768D metrics:

```text
Precision@1   = 0.5476
Precision@3   = 0.2207
Precision@5   = 0.1388
Precision@10  = 0.0740
MAP@100       = 0.6162
```

### Base → FT NDCG comparison used in the end-to-end pipeline

| Dimension | Base | FT | %change |
|---:|---:|---:|---:|
| 768 | 0.523048 | **0.641648** | 22.67486292 |
| 512 | 0.516347 | **0.632601** | 22.51459668 |
| 256 | 0.498844 | **0.621455** | 24.57903284 |
| 128 | 0.468557 | **0.599204** | 27.88272958 |
| 64 | 0.405503 | **0.559806** | 38.05216984 |

- **Average gain % across metrics**---->**27.30714627**
- **Variance of % gain across metrics**-----> **60.6223122**
- **Standard deviation **-----> **7.786033149**

---

## Qualitative Retrieval Sanity Check

### Example for similarity check :-

- Inclusion and exclsuion chunk(1-1 each):-

```text
incl_chunk=''' Absence of suspicion of dementia, based on normal performance according to age and educational level at neuropsychological testing defined as:
Free recall ≥17 and total recall ≥40 for the Free and Cued Selective Reminding Test (Grober and Buschke test 52) MMSE ≥ norm for age and educational level (defined by mean - 1 SD)
Isaac's set test ≥ norm for age and educational level (defined by mean - 1 SD)
Matched to age and gender of the cases
Patient benefiting from social insurance'''

excl_chunk='''History of Parkinson's disease or other neurodegenerative disorder
History of Horton's disease
History of inflammatory neuropathies (in particular Devic's disease, multiple sclerosis)
History of vascular ischemic neuropathies and chronic intracranial hypertension
History of pituitary tumors
Presence of diseases (systemic and/or ocular diseases) or behavioural or cognitive symptoms incompatible with eye examination
Known diabetes
Person under tutorship or curatorship, person unable to express consent'''
```


- Inclusion and exclusion query list(1 each) generated from corresponding inclusion and exclusion sungle chunks :-

``` text
incl_query_list =['''Does the participant demonstrate cognitive performance within the age- and
education-adjusted normal range on the required neuropsychological assessments, with no clinical suspicion of dementia?''',]

excl_query_list=['''Does the participant have any history of neurological, neurodegenerative, vascular, inflammatory,
intracranial, pituitary, or other systemic/ocular disease that could affect study eligibility or the planned eye examination?''',]
```

### Nomic ai fine tune(iteartion 3)  vs base e5 performance:

| Query → Chunk type | Similarity(Nomic ft) | Similarity(base e5) |
|---|---:|---:|---:|
| Inclusion query→ Inclusion chunk | 0.6381 | 0.8611 |
| Exclusion query→ Exclusion chunk | 0.4965 | 0.8957 |
| Inclusion query→ Exclusion chunk | 0.2535 | 0.8200 |
| Exclusion query→ Inclusion chunk | 0.2150 | 0.8034 |

- The relevant query/chunk pairs shows stronger similarity than cross-type irrelevant pairs in case of nomic fine tuned model than **base e5** model(as evident from high similarity scores for dis-similar query-chunk-->row 4th, 5th).
- These raw similarities are used only as a qualitative sanity check. They don't depict real similarity, reason being that the the geometric norms of vector embeddings get scaled up/down while training.
>Infact Retrieval metrics remain the primary evaluation. The same happened with nomic ai while fine tuning(during RAG clinical trial runs for example:- the base had magnified scores due to greter norms, but poor discriminative power as de[picted by metrics).==>
just refer block-9 o/p for **RAG_CT_complete_compressed/ct-rag-pipeline.ipynb**--->even though top retrieved chunk for input question is excatly same, there is divergence of similarity score(ft<base) due to norm vectors/scale decresing during ft. 
---

## Production Retrieval

```mermaid
flowchart LR
    Q[User Question] --> E[Fine-tuned Nomic]
    E --> V[Normalized 768D Vector]
    V --> F[FAISS IndexFlatIP]
    F --> K[Top-4 Chunks]
    K --> G[RAG3 Generator]
```

Because embeddings are normalized:

```text
Inner Product ≈ Cosine Similarity
```

The final retrieval configuration is:

```text
Embedding dimension = 768
Similarity           = Inner Product
Index                = FAISS IndexFlatIP
Top-k                = 4
```

`k=4` was selected as a practical quality/context trade-off rather than as a mathematically optimal value.

---

## Key Decisions

- Fine-tune the **document encoder** rather than relying only on generic embeddings.
- Preserve consolidated trial-level semantics.
- Use Matryoshka training for multi-dimensional representations.
- Use normalized embeddings + `IndexFlatIP`.
- Retain 768D for production retrieval.
- Use top-4 retrieved chunks.
- Treat retrieval improvement as the primary RAG1 objective.

### Takeaway

> RAG1 specializes the retrieval layer so that clinically meaningful trial information is more likely to reach the downstream generator.


### **Datasets**

* [Iteration 1](https://huggingface.co/datasets/vab46/Clinical_trials_anchor-positive-pairs_EmbeddingModel-data)
* [Granular formulation](https://huggingface.co/datasets/vab46/Clinical_trials_anchor-positive-pairs_EmbeddingModel-data2)
* [Final dataset](https://huggingface.co/datasets/vab46/Clinical_trials_anchor-positive-pairs_EmbeddingModel-data_final)

### **Models**

* [Iteration 1](https://huggingface.co/vab46/nomic-embed-text-v1.5_Clinical-Trials_Matryoshka)
* [Granular formulation](https://huggingface.co/vab46/nomic-embed-text-v1.5_Clinical-Trials_Matryoshka2)
* [Final model](https://huggingface.co/vab46/nomic-embed-text-v1.5_Clinical-Trials_Matryoshka_final)