import os
from typing import List, Dict

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from elasticsearch import Elasticsearch, helpers
from elasticsearch import exceptions as es_exceptions

# =========================
# 1. CONFIG & CLIENT SETUP
# =========================

load_dotenv()

ES_HOST = "http://localhost:9200"
ES_INDEX = "documents_rag"   # guna index lain supaya tak kacau yang lama
EMBEDDING_DIM = 1536         # text-embedding-3-small
TOP_K = 3

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
es = Elasticsearch(ES_HOST)


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
# 3. SAMPLE JPM-STYLE DOCUMENTS
# =========================
# Nanti bro boleh ganti ini dengan teks sebenar (garis panduan, pekeliling, dsb.)

documents = [
    {
        "id": "doc-1",
        "title": "Pengenalan Vector Database",
        "content": (
            "Dokumen ini menerangkan konsep asas vector database, "
            "embedding dan kegunaan dalam sistem carian pintar di agensi kerajaan."
        ),
    },
    {
        "id": "doc-2",
        "title": "Elasticsearch untuk Log dan Carian Dokumen",
        "content": (
            "Elasticsearch sesuai digunakan sebagai enjin carian log dan dokumen. "
            "Ia menyokong full-text search, penapisan dan agregasi, serta boleh digabungkan dengan embedding."
        ),
    },
    {
        "id": "doc-3",
        "title": "FAISS untuk Approximate Nearest Neighbour",
        "content": (
            "FAISS ialah pustaka dari Meta untuk approximate nearest neighbour search. "
            "Ia sangat laju dan sesuai untuk carian ke atas jutaan vektor embedding."
        ),
    },
    {
        "id": "doc-4",
        "title": "Hybrid Search untuk Chatbot RAG",
        "content": (
            "Pendekatan hybrid menggabungkan FAISS untuk carian vektor dan Elasticsearch untuk metadata. "
            "Ini sesuai untuk RAG chatbot yang menjawab soalan berdasarkan dokumen dalaman."
        ),
    },
    {
        "id": "doc-5",
        "title": "Scaling Vector Search dalam Persekitaran Kerajaan",
        "content": (
            "Untuk skala besar, kita perlu fikirkan sharding, replication dan pemantauan. "
            "Integrasi dengan sistem sedia ada seperti portal dalaman dan dashboard juga penting."
        ),
    },
]


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
    bulk_ops = []
    for doc in docs:
        bulk_ops.append({
            "_index": ES_INDEX,
            "_id": doc["id"],
            "_source": {
                "title": doc["title"],
                "content": doc["content"],
                # embedding akan diisi kemudian jika perlu
            }
        })

    if bulk_ops:
        helpers.bulk(es, bulk_ops)
        print(f"[OK] {len(bulk_ops)} dokumen dimasukkan ke Elasticsearch.")


# =========================
# 5. FAISS INDEX BUILDING
# =========================

def build_faiss_index(docs: List[Dict]):
    """Build FAISS index from docs' content and return (index, id_map)."""
    base_index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index_with_ids = faiss.IndexIDMap(base_index)

    faiss_ids = []
    vectors = []
    id_map = {}

    print("[INFO] Menjana embedding untuk dokumen...")

    for internal_id, doc in enumerate(docs):
        vec = get_embedding(doc["content"])
        vectors.append(vec)
        faiss_ids.append(internal_id)
        id_map[internal_id] = doc["id"]

    vectors_np = np.vstack(vectors).astype("float32")
    faiss_ids_np = np.array(faiss_ids, dtype="int64")
    index_with_ids.add_with_ids(vectors_np, faiss_ids_np)

    print(f"[OK] FAISS index dibina untuk {len(docs)} dokumen.")
    return index_with_ids, id_map


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
            "score": float(dist),
            "internal_id": int(internal_id),
        })

    return results


# =========================
# 7. GENERATION LAYER (RAG)
# =========================

def answer_question(question: str, retrieved_docs: List[Dict]) -> str:
    """Guna OpenAI untuk jawab soalan berdasarkan dokumen yang ditemui."""
    if not retrieved_docs:
        return "Maaf, saya tidak menjumpai sebarang dokumen yang relevan untuk soalan ini."

    context_parts = []
    for i, doc in enumerate(retrieved_docs, start=1):
        context_parts.append(
            f"[{i}] {doc['title']}\n{doc['content']}"
        )

    context_text = "\n\n".join(context_parts)

    system_prompt = (
        "Anda adalah pembantu teknikal yang membantu menjelaskan konsep-konsep "
        "vector database, Elasticsearch, FAISS dan RAG dalam Bahasa Melayu yang jelas "
        "dan sesuai untuk pegawai kerajaan di Malaysia. Jawapan mestilah ringkas, "
        "tersusun dalam poin jika perlu, dan berdasarkan konteks dokumen yang diberi."
    )

    user_prompt = (
        f"Soalan: {question}\n\n"
        f"Berikut adalah dokumen rujukan:\n\n{context_text}\n\n"
        "Sila berikan jawapan yang padat dan mudah difahami. "
        "Jika sesuai, kaitkan dengan kegunaan di konteks agensi kerajaan/JPM."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    return resp.choices[0].message.content.strip()


# =========================
# 8. MAIN DEMO
# =========================

def main():
    check_es_connection()
    create_es_index()
    index_documents_to_es(documents)
    faiss_index, id_map = build_faiss_index(documents)

    print("\n=== RAG DEMO ===")
    #question = "Macam mana FAISS dan Elasticsearch boleh digabungkan untuk chatbot RAG di agensi kerajaan?"
    #uncommnet utk test jawapan di luar konteks
    question = "Macam mana nak masak ketupat palas?"
    print(f"Soalan: {question}\n")

    retrieved = retrieve_docs(question, faiss_index, id_map, top_k=TOP_K)

    print("Dokumen yang ditemui:")
    for i, doc in enumerate(retrieved, start=1):
        print(f"[{i}] {doc['title']} (id={doc['doc_id']}, distance={doc['score']:.4f})")

    print("\nJawapan model:\n")
    answer = answer_question(question, retrieved)
    print(answer)


if __name__ == "__main__":
    main()
