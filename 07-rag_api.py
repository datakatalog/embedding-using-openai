import os
import glob
from typing import List, Dict, Optional

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from elasticsearch import Elasticsearch, helpers
from elasticsearch import exceptions as es_exceptions
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# =========================
# 1. CONFIG & CLIENT SETUP
# =========================

load_dotenv()

ES_HOST = "http://localhost:9200"
ES_INDEX = "documents_files"   # Sama seperti 06-rag_from_files.py
EMBEDDING_DIM = 1536           # text-embedding-3-small
TOP_K = 3
DATA_FOLDER = "data"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
es = Elasticsearch(ES_HOST)

# Global state untuk API
faiss_index: Optional[faiss.Index] = None
id_map: Optional[Dict[int, str]] = None


# =========================
# 2. EMBEDDING FUNCTION
# =========================

def get_embedding(text: str) -> np.ndarray:
    """Get OpenAI embedding for a piece of text."""
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    vec = resp.data[0].embedding
    return np.array(vec, dtype=np.float32)


# =========================
# 3. LOAD & CHUNK DOKUMEN DARI FOLDER
# =========================

def load_documents_from_folder(folder_path: str,
                               max_chars_per_chunk: int = 1000) -> List[Dict]:
    """
    Baca semua .txt dalam folder, pecah kepada chunk kecil.
    - max_chars_per_chunk: had panjang setiap chunk (lebih kecil = lebih fokus).
    """
    docs: List[Dict] = []

    if not os.path.isdir(folder_path):
        print(f"[WARN] Folder '{folder_path}' tidak wujud. Sila cipta dan letak fail .txt di dalamnya.")
        return docs

    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    if not txt_files:
        print(f"[WARN] Tiada fail .txt dijumpai dalam folder '{folder_path}'.")
        return docs

    print(f"[INFO] Menemui {len(txt_files)} fail .txt dalam '{folder_path}'.")

    doc_counter = 0
    for path in txt_files:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        # Nama file sebagai title asas
        filename = os.path.basename(path)
        base_title = os.path.splitext(filename)[0].replace("_", " ").title()

        # Pecah ikut double newline sebagai "perenggan besar"
        blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
        if not blocks:
            continue

        # Gabung block menjadi chunk yang panjangnya <= max_chars_per_chunk
        current_chunk = ""
        chunk_index = 1

        def flush_chunk(chunk_text: str, idx: int):
            nonlocal doc_counter
            chunk_text = chunk_text.strip()
            if not chunk_text:
                return
            doc_counter += 1
            doc_id = f"{filename}-chunk-{idx}"
            title = f"{base_title} (Bahagian {idx})"
            docs.append({
                "id": doc_id,
                "title": title,
                "content": chunk_text,
                "source_file": filename,
            })

        for block in blocks:
            # Tambah block ke chunk semasa jika masih muat
            if len(current_chunk) + len(block) + 2 <= max_chars_per_chunk:
                if current_chunk:
                    current_chunk += "\n\n" + block
                else:
                    current_chunk = block
            else:
                # flush chunk lama, mula chunk baru
                flush_chunk(current_chunk, chunk_index)
                chunk_index += 1
                current_chunk = block

        # flush chunk terakhir
        flush_chunk(current_chunk, chunk_index)

    print(f"[OK] {doc_counter} chunk dokumen dimuatkan dari folder '{folder_path}'.")
    return docs


# =========================
# 4. ELASTICSEARCH HELPERS
# =========================

def check_es_connection():
    try:
        info = es.info()
        print(f"[OK] Connected to Elasticsearch cluster: {info['cluster_name']}")
    except es_exceptions.ConnectionError:
        print("[ERROR] Elasticsearch tak dapat dicapai di", ES_HOST)
        print("  - Pastikan Docker container 'es-dev' sedang running")
        print("  - Contoh: docker start es-dev")
        raise


def create_es_index():
    """Create index if not exists (idempotent)."""
    if es.indices.exists(index=ES_INDEX):
        print(f"[INFO] Index '{ES_INDEX}' sudah wujud, skip create.")
        return

    mapping = {
        "mappings": {
            "properties": {
                "title":   {"type": "text"},
                "content": {"type": "text"},
                "source_file": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIM,
                    "index": False  # FAISS handle ANN utama
                }
            }
        }
    }
    es.indices.create(index=ES_INDEX, body=mapping)
    print(f"[OK] Index '{ES_INDEX}' dicipta.")


def index_documents_to_es(docs: List[Dict]):
    if not docs:
        print("[WARN] Tiada dokumen untuk diindeks ke Elasticsearch.")
        return

    bulk_ops = []
    for doc in docs:
        bulk_ops.append({
            "_index": ES_INDEX,
            "_id": doc["id"],
            "_source": {
                "title": doc["title"],
                "content": doc["content"],
                "source_file": doc.get("source_file", "unknown"),
            }
        })

    helpers.bulk(es, bulk_ops)
    print(f"[OK] {len(bulk_ops)} dokumen dimasukkan ke Elasticsearch.")


# =========================
# 5. FAISS INDEX BUILDING
# =========================

def build_faiss_index(docs: List[Dict]):
    """Build FAISS index from docs' content and return (index, id_map)."""
    if not docs:
        raise ValueError("Tiada dokumen untuk dibina FAISS index.")

    base_index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index_with_ids = faiss.IndexIDMap(base_index)

    faiss_ids = []
    vectors = []
    local_id_map: Dict[int, str] = {}

    print("[INFO] Menjana embedding untuk dokumen...")

    for internal_id, doc in enumerate(docs):
        vec = get_embedding(doc["content"])
        vectors.append(vec)
        faiss_ids.append(internal_id)
        local_id_map[internal_id] = doc["id"]

    vectors_np = np.vstack(vectors).astype("float32")
    faiss_ids_np = np.array(faiss_ids, dtype="int64")
    index_with_ids.add_with_ids(vectors_np, faiss_ids_np)

    print(f"[OK] FAISS index dibina untuk {len(docs)} dokumen.")
    return index_with_ids, local_id_map


# =========================
# 6. RETRIEVAL LAYER
# =========================

def retrieve_docs(query: str, faiss_index, id_map, top_k: int = TOP_K) -> List[Dict]:
    """Retrieve top_k dokumen paling hampir dengan query."""
    q_vec = get_embedding(query).astype("float32")[None, :]
    distances, internal_ids = faiss_index.search(q_vec, top_k)

    internal_ids_list = internal_ids[0]
    doc_ids = [id_map[i] for i in internal_ids_list if i in id_map]

    if not doc_ids:
        return []

    resp = es.mget(index=ES_INDEX, ids=doc_ids)

    results = []
    for hit, dist, internal_id in zip(resp["docs"], distances[0], internal_ids_list):
        if not hit.get("found"):
            continue
        src = hit["_source"]
        results.append({
            "doc_id": hit["_id"],
            "title": src["title"],
            "content": src["content"],
            "source_file": src.get("source_file", "unknown"),
            "score": float(dist),
            "internal_id": int(internal_id),
        })

    return results


# =========================
# 7. GENERATION LAYER (RAG) DENGAN CITATION
# =========================

def answer_question(question: str, retrieved_docs: List[Dict]) -> str:
    """Guna OpenAI untuk jawab soalan berdasarkan dokumen yang ditemui, dengan citation fail asal."""
    if not retrieved_docs:
        return (
            "Maaf, saya tidak menjumpai sebarang dokumen yang relevan untuk soalan ini. "
            "Sila pastikan dokumen telah dimuat naik ke folder 'data' dan berkaitan dengan soalan."
        )

    # Bina konteks untuk RAG
    context_parts = []
    for i, doc in enumerate(retrieved_docs, start=1):
        context_parts.append(
            f"[{i}] {doc['title']} (fail: {doc['source_file']})\n{doc['content']}"
        )

    context_text = "\n\n".join(context_parts)

    system_prompt = (
        "Anda adalah pembantu teknikal yang menjawab soalan berdasarkan dokumen rasmi "
        "dalam folder 'data'. Jawapan hendaklah dalam Bahasa Melayu yang jelas, "
        "ringkas dan sesuai untuk pegawai kerajaan di Malaysia. "
        "Jika maklumat tidak cukup, nyatakan dengan jujur."
    )

    user_prompt = (
        f"Soalan: {question}\n\n"
        f"Berikut adalah petikan dokumen berkaitan:\n\n{context_text}\n\n"
        "Sila berikan jawapan yang padat, tersusun, dan jika sesuai, gunakan poin bernombor."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    jawapan = resp.choices[0].message.content.strip()

    # Tambah citation di hujung jawapan
    citation_lines = []
    for i, doc in enumerate(retrieved_docs, start=1):
        citation_lines.append(
            f"- [{i}] {doc['title']} (fail: {doc['source_file']}, id: {doc['doc_id']})"
        )

    citation_block = "Rujukan dokumen:\n" + "\n".join(citation_lines)

    return f"{jawapan}\n\n{citation_block}"


# =========================
# 8. FASTAPI SETUP
# =========================

app = FastAPI(
    title="RAG API (Elasticsearch + FAISS + OpenAI)",
    description="API ringkas untuk tanya soalan berdasarkan dokumen dalam folder 'data/'.",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    title: str
    source_file: str
    doc_id: str
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]


@app.on_event("startup")
def startup_event():
    """Inisialisasi sekali sahaja bila API start."""
    global faiss_index, id_map

    print("=== STARTUP: Initializing RAG engine ===")
    check_es_connection()
    create_es_index()

    docs = load_documents_from_folder(DATA_FOLDER, max_chars_per_chunk=1200)
    if not docs:
        print("[WARN] Tiada dokumen dalam folder 'data'. API masih hidup, tapi jawapan mungkin sentiasa kosong.")
        return

    index_documents_to_es(docs)
    faiss_index, id_map = build_faiss_index(docs)
    print("=== STARTUP: RAG engine siap digunakan ===")


@app.get("/health")
def health_check():
    """Endpoint ringkas untuk check API hidup."""
    return {"status": "ok", "faiss_ready": faiss_index is not None}


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):
    """Terima soalan, pulangkan jawapan + senarai sumber dokumen."""
    if faiss_index is None or id_map is None:
        raise HTTPException(
            status_code=503,
            detail="RAG engine belum siap. Pastikan terdapat dokumen dalam folder 'data' dan restart API."
        )

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Soalan tidak boleh kosong.")

    retrieved = retrieve_docs(question, faiss_index, id_map, top_k=TOP_K)
    answer = answer_question(question, retrieved)

    sources = [
        Source(
            title=doc["title"],
            source_file=doc["source_file"],
            doc_id=doc["doc_id"],
            distance=doc["score"],
        )
        for doc in retrieved
    ]

    return AskResponse(answer=answer, sources=sources)
