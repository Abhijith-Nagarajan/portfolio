---
title: "Mental Health Chatbot — RAG over PubMed"
description: "An end-to-end retrieval-augmented generation system that grounds a local LLM in peer-reviewed PubMed abstracts to answer questions on the neurochemistry of mental health, evaluated against the BioASQ-13b biomedical QA benchmark."
date: 2025-05-17
featured: true
weight: 10
metric: "BERTScore 0.67 · Answer Correctness 0.60"
domains: ["Applied AI"]
externalUrl: "https://github.com/Abhijith-Nagarajan/Mental_Health_Chatbot"
tags:
  - "RAG"
  - "LlamaIndex"
  - "FAISS"
  - "BioBERT"
  - "llama.cpp"
  - "LLM Evaluation"
  - "RAGAS"
  - "Biomedical NLP"
  - "Local LLMs"
  - "Python"
---

The project sits at the intersection of AI, psychology, and neuroscience. The chatbot links neurotransmitters (dopamine, serotonin, GABA, glutamate, norepinephrine, acetylcholine, endorphins) to mood disorders (depression, anxiety, bipolar, schizophrenia, OCD, PTSD), and surfaces how nutrition, exercise, and lifestyle affect well-being.

## Architecture

```mermaid
flowchart LR
    subgraph DATA["Data Acquisition"]
        P["PubMed via<br/>NCBI Entrez"] -->|"MeSH queries:<br/>6 disorders × 7 NTs"| J["1,500+ abstracts<br/>(JSON on disk)"]
    end

    subgraph IDX["Indexing"]
        J --> D["LlamaIndex<br/>Documents"]
        D -->|"SentenceSplitter<br/>512 tok / 50 overlap"| C["Chunks"]
        C --> E["BioBERT<br/>embeddings"]
        E --> F["FAISS<br/>IndexFlatL2"]
    end

    subgraph CHAT["Chat Engine"]
        Q["User question"] --> CE["CondenseQuestion<br/>ChatEngine"]
        MB["ChatMemoryBuffer<br/>600 tokens"] -.-> CE
        CE -->|"condensed query"| F
        F -->|"top-k = 3 chunks"| LLM["Phi-2 Orange Q4_K_M<br/>via llama.cpp"]
        LLM --> ANS["Answer"]
    end

    subgraph EVAL["Evaluation"]
        BA["BioASQ-13b<br/>100 questions"] --> R["RAGAS<br/>+ BertScore"]
        ANS --> R
        R --> M["Correctness: 0.60<br/>BertScore: 0.67"]
    end
```

- **Data acquisition** — 1,500+ PubMed abstracts scraped via Biopython Entrez. MeSH queries built from a Cartesian product of 6 disorders (depression, anxiety, bipolar, schizophrenia, OCD, PTSD) × 7 neurotransmitters (dopamine, serotonin, GABA, glutamate, norepinephrine, acetylcholine, endorphins), persisted as JSON.
- **Indexing** — BioBERT embeddings (biomedical NLI/STS fine-tune) feeding a FAISS `IndexFlatL2` vector store; `SentenceSplitter` with 512-token chunks and 50-token overlap.
- **Retrieval + generation** — LlamaIndex `CondenseQuestionChatEngine` with a `ChatMemoryBuffer` for multi-turn follow-ups; local quantized GGUF LLM (Phi-2 Orange Q4_K_M) via `llama.cpp`. No paid APIs at inference time.
- **Evaluation** — RAGAS `AnswerCorrectness` against the first 100 BioASQ-13b training questions and their cited PMIDs, with the same local Phi-2 model wrapped as the judge.

## Key design decisions

- **Local quantized model** for reproducibility and zero API cost. The small context window directly shaped top-k, chunk size, and memory-buffer choices.
- **BioBERT over generic sentence-transformer** — domain match beat model size for jargon-dense PubMed text.
- **FAISS `IndexFlatL2`** (exact search) at sub-1k vectors removes recall variance as a confound when tuning the rest of the pipeline.
- **BioASQ for evaluation** — peer-reviewed biomedical QA benchmark with curated answers, a stronger signal than self-generated ground truth.

## Results

Final evaluation on the BioASQ-13b subset:

- **RAGAS Answer Correctness**: 0.6
- **BertScore**: 0.67

## Open threads

- Hybrid retrieval (BM25 + dense) for rare-term queries like specific drug names and gene IDs.
- Cross-encoder reranker between retrieval and generation.
- Larger or hosted judge model for higher-confidence evaluation.
- Branched/adaptive RAG routing lifestyle vs. mechanism questions to different sub-indexes.

[View on GitHub →](https://github.com/Abhijith-Nagarajan/Mental_Health_Chatbot)
