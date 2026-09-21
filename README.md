# RAG using vLLM with Qwen + ChromaDB (Google Colab, T4 GPU)

A Retrieval-Augmented Generation (RAG) system that serves a **Qwen** model through **vLLM** for fast inference, using **ChromaDB** as a persistent vector store — designed to run end-to-end on a free/standard **Google Colab T4 GPU** instance.

## Overview

This project combines:
- **vLLM** — a high-performance inference engine used to serve the Qwen model with an OpenAI-compatible API, tuned to fit within the memory constraints of a T4 GPU (16 GB VRAM).
- **Qwen** — the LLM used for generation.
- **ChromaDB** — a persistent vector database that stores document embeddings.

The entire pipeline — model serving, embedding, retrieval, and generation — runs inside a single Colab notebook.

## Why Colab + T4?

The T4 GPU is a common free-tier option on Colab, so this setup is tuned accordingly:
- Uses a smaller/quantized Qwen model (e.g. Qwen2.5-1.5B/3B/7B-Instruct, AWQ/GPTQ quantized where needed) to fit T4's 16 GB VRAM.
- Sets conservative vLLM memory and context-length settings (`--gpu-memory-utilization`, `--max-model-len`) to avoid OOM errors.


## Prerequisites

- A Google account with access to Colab (Runtime → Change runtime type → GPU → T4)
- (Optional) Google Drive mounted for persistent ChromaDB storage across sessions

## Setup (Colab)

### 1. Select the T4 runtime

`Runtime → Change runtime type → Hardware accelerator → T4 GPU`

### 2. Install dependencies

```python
!pip install -q vllm chromadb sentence-transformers openai
```

### 3. Start the vLLM server

Launched in the background within the notebook so the same cell/session can query it:

```python
import subprocess, time, requests

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"   # sized for T4 VRAM
VLLM_PORT = 8001

vllm_process = subprocess.Popen([
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_NAME,
    "--port", str(VLLM_PORT),
    "--gpu-memory-utilization", "0.85",
    "--max-model-len", "4096",
    "--dtype", "half",
])

### 4. Ingest documents into ChromaDB


import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = client.get_or_create_collection("docs")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# chunk_texts, chunk_ids, chunk_metadatas prepared from your source docs
embeddings = embedder.encode(chunk_texts).tolist()
collection.add(
    documents=chunk_texts,
    embeddings=embeddings,
    ids=chunk_ids,
    metadatas=chunk_metadatas,
)
```

### 5. Query the RAG pipeline

```python
from openai import OpenAI

client_llm = OpenAI(base_url=f"http://localhost:{VLLM_PORT}/v1", api_key="not-needed")

def rag_query(question, top_k=4):
    q_emb = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=top_k)
    context = "\n\n".join(results["documents"][0])

    prompt = f"Answer the question using only the context below.\n\nContext:\n{context}\n\nQuestion: {question}"
    response = client_llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

print(rag_query("What does the document say about deployment steps?"))
```

## Project Structure

```
.
├── RAG_vLLM_Qwen_Colab.ipynb   # Main Colab notebook (setup, ingest, query)
├── documents.py                        # Source documents for ingestion
├── chroma_store/                   # Local fallback persistent store (if Drive not mounted)
└── README.md
```


