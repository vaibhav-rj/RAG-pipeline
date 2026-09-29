# RAG3 — Clinical-Trial Answer Generation with QLoRA

## Objective

RAG1 improves **what information is retrieved**.

RAG3 improves **how the retrieved information is converted into an answer**.

```text
RAG1 → Retrieval specialization
RAG3 → Generation specialization
```

The objective was to reduce unsupported generation, improve completeness, preserve clinical-trial-specific numbers/criteria, and produce concise context-grounded answers.

---

## Overall Architecture

```mermaid
flowchart LR
    Q[Question] --> R[RAG1 Fine-tuned Nomic]
    R --> F[FAISS Top-4 Retrieval]
    F --> C[Retrieved Context]
    C --> L[LoRA Llama 3.1 8B]
    L --> A[Grounded Answer]

    T[Qwen 2.5 7B Teacher] --> D[Pseudo Reference Answers]
    D --> L
```

The Qwen teacher is used **offline for dataset construction only**.

It is not part of final inference.

---

# 1. Teacher Reference Generation

## Teacher Model

```text
Qwen 2.5 7B Instruct
Temperature = 0.1
```

Multiple questions were supplied with their associated clinical-trial context.

The teacher was instructed to:

- use only explicit context,
- avoid outside knowledge,
- avoid unsupported inference,
- preserve numbers and eligibility constraints,
- answer only the relevant question,
- remain concise but sufficiently complete,
- return exactly one answer per question,
- produce valid JSON,
- keep each question tied to its associated context.

---

## Eligibility-Specific Rules

Eligibility questions required additional care.

### Rule 14

Include:

> relevant inclusion/exclusion criteria and key thresholds needed to support the answer

rather than dumping every criterion from the trial.

### Rule 15

For questions such as:

```text
Could this patient qualify?
Is the patient eligible?
Does the patient meet the criteria?
```

the answer must distinguish between:

```text
Criteria explicitly satisfied by the question
            +
Additional criteria that still need to be satisfied
```

### Rule 16

Each question must be answered strictly against its **own associated context**.

No eligibility criteria or conclusions should be transferred from another semantically similar question.

---

# 2. Teacher Generation Failure Analysis

The initial strict teacher prompt sometimes produced truncated JSON, particularly when four answers had to include extensive eligibility criteria.

The main constraint was:

```text
num_predict = 300
```

### Fix

Generation was retried up to three times:

```text
300 → 375 → 450 tokens
```

Only successfully generated groups were written to the production checkpoint/completed set.

Failed groups remained eligible for retry.

Approximately:

```text
6 unique trial/context groups
× 4 anchors
≈ 24 failed records
```

remained after the initial retry process.

---

# 3. Reference-Answer Cleaning

Some successful teacher outputs were technically valid JSON but poor supervision examples.

Examples included:

```text
45
```

instead of a complete grounded answer such as:

```text
The age should be 45 years, provided the relevant inclusion criteria are met.
```

Other problematic outputs included:

```text
True
False
NaN
```

for eligibility/qualification questions.

These were normalized into explicit, context-grounded answers where possible.

This improved the training signal for:

- eligibility questions,
- qualification questions,
- age/temporal questions,
- boolean questions.

---

# 4. Candidate Model

## Base Model

```text
Llama 3.1 8B Instruct
```

## Fine-Tuning

The candidate was specialized using:

> **QLoRA**

Conceptually:

```text
W' = W + ΔW
ΔW = BA
```

where:

- `W` = frozen pretrained weights
- `A/B` = trainable low-rank adapter matrices
- `ΔW` = learned task-specific update

QLoRA additionally loads the base model weights in **4-bit quantized form**, reducing GPU memory requirements.

---

## LoRA Configuration

The adapter targeted:

```text
all-linear
```

Important parameters:

- LoRA rank controls adapter capacity.
- LoRA alpha controls adapter scaling.
- The pretrained model remains frozen.
- Only adapter parameters are trained.

Gradient checkpointing was used to reduce activation memory.

---

# 5. Candidate Training Data

Each training example contained:

```text
anchor
+
retrieved/context chunk
+
reference answer
```

The full chat-formatted prompt was tokenized.

Loss masking ensured that:

```text
Prompt/context → -100
Assistant answer → training labels
```

Therefore the model was trained primarily on generating the desired answer rather than reproducing the prompt.

EOS was used as the padding token.

---

# 6. Candidate Prompt

The candidate instruction was deliberately aligned with the teacher's grounding requirements:

```text
You are a clinical-trial question answering assistant.

Answer the user's QUESTION using the provided CONTEXT.

- Answer directly and naturally.
- Base the answer on the CONTEXT and do not introduce unsupported trial-specific facts.
- Synthesize relevant information rather than simply copying the context.
- Include important numbers, thresholds, dates, age ranges, conditions, interventions, and eligibility criteria when relevant.
- For eligibility questions, explain whether the information given satisfies the relevant criteria and mention other important criteria when relevant.
- Do not include irrelevant trial information.
- If the CONTEXT does not provide enough information, say so clearly.
- Preserve trial-specific details accurately.
- Answer concisely, factually, and sufficiently completely.
```

The strengthened eligibility instruction was intended to address the earlier Rule-15-style weakness.

---

# 7. RAG3 Iteration 1

## Training Configuration

```text
Epochs                    = 1
Per-device batch size     = 2
Gradient accumulation     = 8
Effective batch size      = 16
Learning rate             = 2e-4
Scheduler                 = cosine
Weight decay              = 0.01
Gradient checkpointing    = enabled
Precision                 = FP16
Optimizer                 = Adam8bit
```

Training was constrained by the available GPU memory/runtime.

---

## Iteration-1 Results

| Metric | Base | LoRA | Δ |
|---|---:|---:|---:|
| ROUGE-L | 0.689654 | **0.691370** | +0.001716 |
| BERTScore-F1 | 0.945721 | **0.946055** | +0.000334 |
| Numeric Groundedness | **0.976151** | 0.971657 | -0.004494 |
| Numeric Recall | **0.903712** | 0.897009 | -0.006703 |

### Paired comparison

**ROUGE-L**

```text
Tie       = 351
LoRA wins = 227
Base wins = 205
```

**BERTScore**

```text
LoRA wins = 272
Base wins = 261
Tie       = 250
```

### Interpretation

Iteration 1 produced **near-parity with a modest LoRA advantage on text-similarity metrics**, but not a dramatic generation improvement.

Possible limiting factors included:

- only one training epoch,
- small per-device batch,
- imperfect pseudo-reference answers,
- failed teacher generations,
- filtered scalar/NaN references,
- mismatch between strict teacher instructions and the candidate prompt.

---

# 8. RAG3 Iteration 2

## Motivation

Iteration 1 exposed several weaknesses:

```text
Teacher references
      ↓
NaN / boolean / integer-only answers
      ↓
weak supervision
```

and:

```text
Candidate prompt
      ↓
less explicit eligibility qualification behavior
```

The second iteration therefore targeted:

1. reference-answer normalization,
2. recovery/handling of failed references,
3. stronger eligibility instructions,
4. longer training,
5. two-epoch training, larger per-device batch size, improved reference cleaning, and strengthened eligibility instructions.

---

## Changes

| Component | Iteration 1 | Iteration 2 |
|---|---:|---:|
| Epochs | 1 | **3** |
| Per-device batch | 2 | **4** |
| Gradient accumulation | 8 | 4 |
| Effective batch | 16 | **16** |
| Reference cleaning | Initial | **Improved** |
| Eligibility prompt | Initial | **Strengthened** |

The improvements were intentionally combined because the goal was to improve the overall supervision/training pipeline rather than perform a single-variable ablation.

Therefore, the resulting improvement **cannot be attributed solely to increasing epochs or batch size**.

---

# 9. Iteration-2 Results

## Epoch 1
- The results of iteration-2 epoch 1 have been neglected as it were quite inferior for most entities than base-model.


## Epoch 2 — Final Iteration-2 Checkpoint

### Epoch 2 Question-Type Analysis

| Question type | n | Base ROUGE | LoRA ROUGE | Base BERT | LoRA BERT |
|---|---:|---:|---:|---:|---:|
| Eligibility / Qualification | 228 | 0.635210 | **0.655508** | 0.927499 | **0.932038** |
| Study Design | 227 | **0.823676** | 0.808107 | **0.967911** | 0.967096 |
| Other | 180 | 0.595429 | **0.598936** | **0.934205** | 0.934145 |
| Intervention | 64 | 0.662192 | **0.673651** | 0.943153 | **0.944862** |
| Temporal | 48 | **0.582274** | 0.569226 | **0.933253** | 0.930462 |
| Numeric / Threshold | 37 | **0.682030** | 0.660298 | **0.942677** | 0.938950 |

The strongest targeted improvement occurred in: **Eligibility / Qualification** followed by **Intervention**.

### Epoch 2 Paired Evaluation

-  ROUGE-L

```text
Tie       = 365
LoRA wins = 211
Base wins = 208
```

-  BERTScore

```text
LoRA wins = 275
Base wins = 268
Tie       = 241
```
The paired results again indicate **near-parity rather than a dramatic transformation**.

### Epoch 2 Grounding & Completeness

| Metric | Base | LoRA |
|---|---:|---:|
| Mean context-support score | 0.442570 | 0.441146 |
| Low-support sentence rate | 0.022534 | 0.022534 |
| Numeric groundedness | 0.968105 | 0.965843 |
| Numeric recall | 0.900501 | **0.903364** |

### Epoch 2 OVERALL RESULTS

| Metric | Base | LoRA | Absolute Δ |
|---|---:|---:|---:|
| ROUGE-L | 0.681817 | **0.683128** | +0.001311 |
| BERTScore-F1 | 0.943086 | **0.943949** | +0.000863 |
| Numeric Groundedness | **0.968105** | 0.965843 | -0.002261 |
| Numeric Recall | 0.900501 | **0.903364** | +0.002864 |

Overall generation remained **very close to the Base model**, with small improvements in ROUGE-L(*0.635210 → 0.655508*), BERTScore(*0.927499 → 0.932038*) 
and numeric recall(0.851973 → 0.871438), while numeric groundedness decreased slightly(*0.919022 → 0.907919*). WHile context-support and low-support sentence rate were stagnant.


## Epoch-3 — Final Iteration-2 Checkpoint

A third epoch was run from the Iteration-2 configuration using the same seed/data split. Epoch 3 produced the strongest overall standalone Iteration-2 results and was retained as the generation checkpoint for the current final CT-RAG pipeline.

---

### Epoch-3 Question-Type Results

| Question type | n | Base ROUGE | LoRA ROUGE | Base BERT | LoRA BERT |
|---|---:|---:|---:|---:|---:|
| Eligibility / Qualification | 228 | 0.656928 | **0.657644** | 0.932666 | **0.933110** |
| Intervention | 64 | 0.661598 | **0.667439** | 0.943098 | **0.945551** |
| Numeric / Threshold | 37 | 0.663168 | **0.669361** | 0.939437 | **0.941543** |
| Other | 180 | 0.594660 | **0.609685** | 0.934415 | **0.937659** |
| Study Design | 227 | 0.809468 | **0.818702** | 0.965400 | **0.967680** |
| Temporal | 48 | **0.616053** | 0.595652 | **0.935034** | 0.932568 |

Epoch 3 improved ROUGE/BERTScore across most slices, while **Temporal remained difficult** and did not improve on these text-similarity metrics.

### Epoch 3 Paired Evaluation

-  ROUGE-L

```text
Tie       = 375
LoRA wins = 216
Base wins = 193
```

-  BERTScore

```text
LoRA wins = 275
Base wins = 268
Tie       = 241
```
The paired results again indicate **far more contrast in favour of LoRA than Base**.

### Epoch-3 Grounding Diagnostics

| Metric | Base | LoRA |
|---|---:|---:|
| Mean context-support | 0.443536 | **0.443660** |
| Low-support sentence rate | 0.018495 | 0.019770 |
| Numeric groundedness | 0.964547 | **0.967221** |
| Numeric recall | 0.891752 | **0.894615** |

### Epoch-3 OVERALL RESULTS

| Metric | Base | LoRA — Epoch 3 | Absolute Δ |
|---|---:|---:|---:|
| ROUGE-L | 0.684971 | **0.690823** | +0.005852 |
| BERTScore-F1 | 0.943861 | **0.945544** | +0.001683 |
| Numeric Groundedness | 0.964547 | **0.967221** | +0.002674 |
| Numeric Recall | 0.891752 | **0.894615** | +0.002863 |

Both Rouge-L registered moderate gains while BERTScore-F1 had marginal gains. Context support was essentially unchanged, while numeric groundedness and numeric recall both showed small positive LoRA deltas. 
Therefore Epoch 3 provides evidence of improved overall generation/numeric behaviour, but not a broad independent grounding breakthrough.

## Epoch 2 → Epoch 3 Comparison

Metric	Epoch 2 FT−Base	Epoch 3 FT−Base	Change in Δ
ROUGE-L	+.001311	+.005852	+.004541
BERTScore-F1	+.000863	+.001683	+.000820
Numeric groundedness	−.002261	+.002674	+.004935
Numeric recall	+.002864	+.002863	~0


>Epoch 3 improved the LoRA-vs-Base delta on all four overall metrics relative to Epoch 2. The most notable change was ROUGE-L, where the LoRA advantage increased from +0.00131 to +0.00585. Numeric groundedness also 
changed from a small LoRA deficit (−0.00226) in Epoch 2 to a positive LoRA delta (+0.00267) in Epoch 3. Numeric recall, however, was effectively unchanged at +0.00286.

>**These results support retaining Epoch 3 as the stronger Iteration-2 checkpoint under the same seed/data split, but they do not isolate the causal effect of the third epoch. Iteration 2 already incorporated reference cleaning, prompt strengthening and training-configuration changes.**


### Training-Loss Behaviour

>Across Iteration-2 training, loss showed an early/mid low-loss region followed by a late rise. Epoch 3 followed the same broad pattern, reaching approximately 0.10 through the earlier/middle region before rising toward ~0.14 near the end. This recurring pattern was treated as a training-dynamics limitation rather than evidence that later batches were uniformly more difficult.


## Epoch 4 — Not Retained

Epoch 4 was evaluated as a final optimization attempt but was not retained because its absolute LoRA ROUGE-L, BERTScore and context-support were below Epoch 3. The final RAG3 checkpoint therefore remains Iteration 2 — Epoch 3, seed 42.

##Retained checkpoint

```text
Llama 3.1 8B Instruct + QLoRA
RAG3 Iteration 2 — Epoch 3
Seed = 42
```

---

# 10. End-to-End RAG Role

RAG3 sits after RAG1 retrieval:

```mermaid
flowchart LR
    Q[User Question]
    Q --> E[Fine-tuned Nomic]
    E --> F[FAISS Top-4]
    F --> C[Retrieved Clinical Context]
    C --> L[QLoRA Llama 3.1]
    L --> A[Grounded Answer]
```

The complete conceptual division is:

```text
RAG1
  ↓
Find the right evidence

RAG2
  ↓
Test alternative query-document alignment
  ↓
Rejected from production

RAG3
  ↓
Generate a concise, complete answer from retrieved evidence
```

---

# 11. Final Experimental Story

The RAG3 work evolved through a sequence of identifiable problems:

```text
Generic LLM
    ↓
Potential hallucination / omission
    ↓
Teacher-generated grounded references
    ↓
Teacher generation failures
    ↓
Retry + improved generation budget
    ↓
Poor scalar / boolean / NaN supervision
    ↓
Reference normalization
    ↓
Eligibility answers still insufficiently structured
    ↓
Stronger candidate instructions
    ↓
Iteration-2 training
    ↓
Epoch 3: strongest Iter-2 standalone checkpoint
    ↓
End-to-end CT-RAG evaluation
```

This is an important part of the project's engineering story: the model was not simply trained once and evaluated. The data-generation and supervision pipeline itself was iteratively debugged.

---

# 12. Limitations

- Teacher references are **pseudo-ground truth**, not expert annotations.
- Some teacher generations initially failed/truncated.
- Training and data-cleaning changes were combined in Iteration 2, so causal attribution is limited.
- Overall generation gains remain modest.
- Numeric groundedness did not improve consistently.
- Some question types remained stronger with the Base model.
- The retained Epoch-3 generator has now been evaluated end-to-end; Epoch 4 was evaluated but not retained because it did not improve the overall checkpoint.

---

# 13. Final Design

```text
                 OFFLINE
┌──────────────────────────────────────┐
│ Qwen 2.5 7B Instruct                 │
│ Context + Questions                  │
│            ↓                         │
│ Grounded Reference Answers           │
└──────────────────┬───────────────────┘
                   ↓
              QLoRA Training
                   ↓
┌──────────────────────────────────────┐
│ Llama 3.1 8B Instruct + LoRA         │
└──────────────────┬───────────────────┘
                   │
                   │ FINAL INFERENCE
                   ↓
Question → RAG1 Retrieval → Top-4 Context
                              ↓
                     LoRA Llama 3.1
                              ↓
                       Final Answer
```

### Takeaway

> RAG3 specialized the generation layer using context-grounded pseudo-references and QLoRA. Iteration 1 established near-parity with the pretrained generator, while Iteration 2 improved eligibility/qualification behavior after targeted reference cleaning, prompt 
strengthening, and longer/larger-batch training. The retained Epoch-3 model has now been evaluated in the complete CT-RAG pipeline.

### **Datasets**
* [Iteration 1](https://huggingface.co/datasets/vab46/Clinical_trials_anchor-contextORpositive-ground-truth_LLM_LORA_ft)
* [Iter2/Final dataset](https://huggingface.co/datasets/vab46/Clinical_trials_anchor-contextORpositive-ground-truth_LLM_LORA-junk_handled_ft)

### **Models**

* [Iteration 1](https://huggingface.co/vab46/llama-3.1-8b-instruct-lora-clinical_iter1)
* [Iter2/Final model](https://huggingface.co/vab46/llama-3.1-8b-instruct-lora-clinical_iter2_epoch3)