# RAG-pipeline
It relates with implementation of RAG pipelines. Quest to fine tune different elements(retrieval , generation broadly) of a domain specific RAG and analyze change from base results. Further experiments on RAG frameworks, prompt enineering and more. 

## <u>Project work brief across thematic folders</u>=>

### 1. Initial RAG Pipeline

Hands-on implementation of end-to-end **RAG pipelines**, covering dense embeddings, FAISS/custom retrieval, Top-K context construction and LLM-based generation using models including **MiniLM, BGE, FLAN-T5 and Llama**. This work established the foundation for subsequent domain-specific RAG optimization.


### 2. Clinical-Trials Domain-Specific RAG Pipeline

Built and iteratively optimized a **domain-specific Clinical-Trials RAG pipeline** across retrieval, query-document alignment and generation. The pipeline evolved through **RAG1–RAG3**: fine-tuned Nomic embeddings using **MNRL + Matryoshka** for CT-specific retrieval; evaluated specialized query encoders and 
asymmetric retrieval in RAG2; and adapted **Llama 3.1 8B Instruct with QLoRA** for context-grounded answer generation in RAG3, using Qwen 2.5 7B Instruct for offline reference-answer generation. The final pipeline integrates **fine-tuned retrieval + LoRA-based generation**, with systematic evaluation of retrieval 
and end-to-end answer quality against the corresponding base pipeline.