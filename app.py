from fastapi import FastAPI
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import httpx
import os

app = FastAPI(title="Policy-Based RAG API")

PERSIST_DIR = "chroma_db"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

if not os.path.exists(PERSIST_DIR):
    raise RuntimeError(f"Vector store '{PERSIST_DIR}' tidak ditemukan. jlankan 'python ingest.py' dulu.")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

class ChatRequest(BaseModel):
    query: str

SYSTEM_PROMPT = """Anda adalah asisten AI yang menjawab pertanyaan berdasarkan kebijakan perusahaan.

ATURAN:
1. Jawab HANYA berdasarkan konteks dokumen di bawah
2. Jika informasi TIDAK ADA dalam konteks, jawab: "Saya tidak menemukan informasi terkait dalam kebijakan perusahaan."
3. Jawab SINGKAT dan LANGSUNG (maksimal 2-3 kalimat)
4. JANGAN menambahkan informasi di luar konteks

Konteks dokumen:
{context}

Pertanyaan: {question}

Jawaban singkat:"""

async def generate_with_ollama(prompt: str, model: str = "gemma2:2b") -> str:
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:  
            response = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": model, 
                    "prompt": prompt, 
                    "stream": False,
                    "options": {
                        "num_predict": 200,      
                        "num_ctx": 4096,         
                        "temperature": 0.2,      
                        "top_p": 0.85,           
                        "repeat_penalty": 1.1,   
                    }
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
    except Exception as e:
        return f"Error: {str(e)}"

@app.get("/")
async def root():
    return {
        "message": "Policy-Based RAG API",
        "status": "running",
        "ollama_host": OLLAMA_HOST,
        "model": "gemma2:2b"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "gemma2:2b"}

@app.post("/chat/policy")
async def chat_policy(request: ChatRequest):
    query = request.query.strip()

    if not query:
        return {"error": "Query tidak boleh kosong."}

    results = db.similarity_search(query, k=2)
    
    if not results:
        return {"answer": "Tidak ditemukan konteks kebijakan yang relevan."}

    context = "\n".join([
        f"Dokumen {i+1}: {r.page_content[:300]}" 
        if len(r.page_content) > 300 
        else f"Dokumen {i+1}: {r.page_content}"
        for i, r in enumerate(results)
    ])

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=SYSTEM_PROMPT
    ).format(context=context, question=query)

    answer = await generate_with_ollama(prompt)

    return {
        "query": query,
        "answer": answer,
        "context_used": [r.metadata for r in results],
        "model": "gemma2:2b"
    }