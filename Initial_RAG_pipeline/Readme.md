# Initial RAG Pipeline Implementations

## Overview

Hands-on implementations of **Retrieval-Augmented Generation (RAG)** pipelines to understand the interaction between dense embeddings, vector retrieval, context construction and LLM-based generation.

The implementations establish the foundation for subsequent **domain-specific RAG optimization**.

## Pipeline

```text
Documents
    ↓
Chunking / Preprocessing
    ↓
Dense Embedding
    ↓
Vector Retrieval
    ↓
Top-K Context
    ↓
Prompt Construction
    ↓
LLM Generation
    ↓
Response
```

## Components Explored

**Embedding models**

* `all-MiniLM-L6-v2`
* `bge-base-en-v1.5`

**Retrieval**

* FAISS
* Customized retrieval components

**Generation**

* `google/flan-t5-small`
* `Llama-3.2-1B-Instruct-GGUF`

## Key Hands-on Work

* Implemented the complete retrieval → generation workflow.
* Experimented with dense semantic representations and similarity-based retrieval.
* Built and tested FAISS-based vector search.
* Explored custom retrieval logic and Top-K context selection.
* Integrated retrieved context with generative LLMs.
* Investigated how retrieval quality influences downstream generation.

## Purpose

This folder represents the **initial RAG implementation stage** of the project.

The subsequent `02_CT_Embedding_Finetuning` folder extends this work by adapting the embedding component specifically for **Clinical-Trials retrieval**.

## Tech Stack

Python · PyTorch · Hugging Face Transformers · Sentence Transformers · FAISS · Jupyter/Colab
