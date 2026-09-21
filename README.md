# RAG on vLLM with Qwen + ChromaDB (Google Colab, T4 GPU)

A Retrieval-Augmented Generation (RAG) system that serves a **Qwen** model through **vLLM** for fast inference, using **ChromaDB** as a persistent vector store — designed to run end-to-end on a free/standard **Google Colab T4 GPU** instance.

## Overview

This project combines:
- **vLLM** — a high-performance inference engine used to serve the Qwen model with an OpenAI-compatible API, tuned to fit within the memory constraints of a T4 GPU (16 GB VRAM).
- **Qwen** — the LLM used for generation.
- **ChromaDB** — a persistent vector database that stores document embeddings so the index survives across Colab sessions (when backed by Google Drive).

The entire pipeline — model serving, embedding, retrieval, and generation — runs inside a single Colab notebook.

## Why Colab + T4?

The T4 GPU is a common free-tier option on Colab, so this setup is tuned accordingly:
- Uses a smaller/quantized Qwen model (e.g. Qwen2.5-1.5B/3B/7B-Instruct, AWQ/GPTQ quantized where needed) to fit T4's 16 GB VRAM.
- Sets conservative vLLM memory and context-length settings (`--gpu-memory-utilization`, `--max-model-len`) to avoid OOM errors.
- Persists ChromaDB data to Google Drive so embeddings don't need to be regenerated every time the runtime resets.

## Architecture

```
┌─────────────┐      ┌──────────────────┐      ┌────────────────────────┐
│   User query │ ───▶ │  Retriever        │ ───▶ │  ChromaDB                │
└─────────────┘      │  (embed + search) │      │  (persisted to Drive)    │
                      └──────────────────┘      └────────────────────────┘
                               │
                               ▼
                      ┌──────────────────┐      ┌────────────────────────┐
                      │  Prompt builder   │ ───▶ │  vLLM server              │
                      │  (query + context)│      │  serving Qwen on T4 GPU   │
                      └──────────────────┘      └────────────────────────┘
                                                          │
                                                          ▼
                                                  ┌────────────────────────┐
                                                  │  Generated answer        │
                                                  └────────────────────────┘
```

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

### 3. Mount Google Drive (for persistent storage)

```python
from google.colab import drive
drive.mount('/content/drive')

CHROMA_PERSIST_DIR = "/content/drive/MyDrive/rag_project/chroma_store"
```

### 4. Start the vLLM server

Launched in the background within the notebook so the same cell/session can query it:

```python
import subprocess, time, requests

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"   # sized for T4 VRAM
VLLM_PORT = 8000

vllm_process = subprocess.Popen([
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_NAME,
    "--port", str(VLLM_PORT),
    "--gpu-memory-utilization", "0.85",
    "--max-model-len", "4096",
    "--dtype", "half",
])

# Wait for the server to become healthy
for _ in range(60):
    try:
        if requests.get(f"http://localhost:{VLLM_PORT}/health").status_code == 200:
            print("vLLM server is up")
            break
    except requests.exceptions.ConnectionError:
        time.sleep(5)
```

> **T4 tip:** if you hit an out-of-memory error, lower `--gpu-memory-utilization`, reduce `--max-model-len`, or switch to a smaller/quantized Qwen checkpoint (e.g. an AWQ variant).

### 5. Ingest documents into ChromaDB

```python
import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = client.get_or_create_collection("docs")

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

# chunk_texts, chunk_ids, chunk_metadatas prepared from your source docs
embeddings = embedder.encode(chunk_texts).tolist()
collection.add(
    documents=chunk_texts,
    embeddings=embeddings,
    ids=chunk_ids,
    metadatas=chunk_metadatas,
)
```

### 6. Query the RAG pipeline

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
├── docs/                          # Source documents for ingestion
├── chroma_store/                   # Local fallback persistent store (if Drive not mounted)
└── README.md
```

## Notes & Limitations

- Colab sessions are ephemeral — GPU memory and the vLLM process are lost on disconnect; only Drive-backed ChromaDB data persists.
- T4 has no support for some newer quantization kernels available on Ampere+ GPUs, so stick to `half`/`float16` dtype or AWQ/GPTQ builds tested on Turing architecture.
- For longer-running or production use, consider a dedicated GPU instance instead of Colab.

## Roadmap

- [ ] Add automatic model size selection based on available VRAM
- [ ] Add a Gradio UI cell for interactive querying
- [ ] Support resuming ingestion from Drive without re-embedding unchanged docs

## License

MIT

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm)
- [Qwen](https://github.com/QwenLM/Qwen)
- [ChromaDB](https://github.com/chroma-core/chroma)
