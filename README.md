# Policy-Based RAG API

## Konsep Arsitektur

```
Dokumen Kebijakan (.md)
         ↓
    ingest.py (Chunking & Embedding)
         ↓
    ChromaDB (Vector Database)
         ↓
    FastAPI Service (/chat/policy)
         ↓
    Ollama (LLM - Mistral)
         ↓
    Response dengan kutip sumber
```

Sistem RAG ini mengambil dokumen kebijakan, memotongnya menjadi chunks dengan konteks Markdown headers, lalu menyimpannya sebagai vector embeddings. Ketika user mengirim query, sistem mencari dokumen paling relevan dan menggunakan LLM untuk generate jawaban dengan menyebutkan sumber.

## Konfigurasi Worker: 4 Worker vs 1 Worker

### 1 Worker (Development/Single Request)

```
Request 1 → Worker 1 → Tunggu Ollama (3-5 detik) → Response
Request 2 → Antre     → Tunggu selesainya Request 1 → Response
Request 3 → Antre     → Tunggu Request 1 & 2 → Response
```

Hanya 1 request bisa dihandle secara concurrent. Request lain harus menunggu.

### 4 Workers (Production/Multiple Concurrent Requests)

```
Request 1 → Worker 1 → Ollama (3-5 detik)
Request 2 → Worker 2 → Ollama (3-5 detik)
Request 3 → Worker 3 → Ollama (3-5 detik)
Request 4 → Worker 4 → Ollama (3-5 detik)
```

Ke-4 request bisa dihandle secara parallel tanpa saling menunggu.

### Alasan Menggunakan 4 Worker

1. **Workload I/O-Bound**: FastAPI menunggu response dari Ollama (LLM generation), bukan compute-intensive. Waktu tunggu tersebut bisa digunakan worker lain untuk handle request berbeda. Dengan 1 worker, resource terbuang karena blocking.

2. **GPU-Bound pada Ollama saja**: Ollama (LLM generation) yang GPU-bound, bukan FastAPI. FastAPI hanya pass-through layer. Lebih banyak worker FastAPI = lebih banyak request concurrent yang bisa diproses, bukan berarti lebih banyak GPU usage (GPU usage tetap di Ollama).

3. **Throughput Meningkat**: Dengan 4 worker, throughput bisa naik 3-4x (asumsi) dibanding 1 worker, terutama di peak load dengan multiple concurrent requests.


### Konfigurasi di Dockerfile & Docker Compose

Dockerfile:
```dockerfile
CMD ["gunicorn", "app:app", "-k", "uvicorn.workers.UvicornWorker", "--workers", "4", "--bind", "0.0.0.0:8000"]
```

