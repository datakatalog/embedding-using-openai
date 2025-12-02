import faiss
import numpy as np
from elasticsearch import Elasticsearch, helpers

# =========================
# 1. CONFIG
# =========================

ES_HOST = "http://localhost:9200"
ES_INDEX = "documents"
EMBEDDING_DIM = 1536       # ikut model embedding bro
TOP_K = 5                  # berapa result nak keluar

# =========================
# 2. SAMBUNG ELASTICSEARCH
# =========================

es = Elasticsearch(ES_HOST)


# =========================
# 3. FUNGSI DUMMY EMBEDDING
# =========================
# NOTE:
# - Tukar fungsi ni kepada OpenAI / HF / apa-apa model sebenar.
# - Di sini saya buat random vector untuk contoh.

import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # ← load .env automatically

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not client:
    raise Exception("Please set OPENAI_API_KEY in .env file")
EMBEDDING_DIM = 1536  # ikut model text-embedding-3-small

def get_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    vec = resp.data[0].embedding
    return np.array(vec, dtype=np.float32)

# =========================
# 4. BUAT SAMPLE DOKUMEN
# =========================

documents = [
    {"id": "doc-1", "title": "Pengenalan Vector Database", "content": "Ini penerangan ringkas tentang vector DB."},
    {"id": "doc-2", "title": "Elasticsearch untuk Log dan Search", "content": "Bagaimana guna Elasticsearch dalam sistem enterprise."},
    {"id": "doc-3", "title": "FAISS untuk ANN", "content": "FAISS digunakan untuk approximate nearest neighbour search."},
    {"id": "doc-4", "title": "Hybrid Search RAG", "content": "Gabungan keyword dan semantic search untuk chatbot."},
    {"id": "doc-5", "title": "Scaling Vector Search", "content": "Teknik scaling, sharding dan replication untuk vector search."},
]

# =========================
# 5. BUAT INDEX ELASTICSEARCH
# =========================

from elasticsearch import exceptions as es_exceptions

def check_es_connection():
    try:
        info = es.info()
        print(f"[OK] Connected to Elasticsearch cluster: {info['cluster_name']}")
    except es_exceptions.ConnectionError:
        print("[ERROR] Elasticsearch tak dapat dicapai di http://localhost:9200")
        print("  - Pastikan Docker container 'es-dev' sedang running")
        print("  - Contoh start semula: docker start es-dev")
        raise


def create_es_index():
    if es.indices.exists(index=ES_INDEX):
        es.indices.delete(index=ES_INDEX)

    mapping = {
        "mappings": {
            "properties": {
                "title":   {"type": "text"},
                "content": {"type": "text"},
                # optional: simpan vector juga di ES untuk backup
                "embedding": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIM,
                    "index": False  # kita guna FAISS untuk ANN utama
                }
            }
        }
    }
    es.indices.create(index=ES_INDEX, body=mapping)


# =========================
# 6. INDEXING: ES + FAISS
# =========================

def build_faiss_index(docs):
    """
    - Kira embedding
    - Simpan ke ES
    - Masukkan vector ke FAISS dengan ID numeric
    """
    # Kita guna IndexFlatL2 (exact); untuk production boleh tukar ke HNSW/IVF etc.
    index = faiss.IndexFlatL2(EMBEDDING_DIM)

    # Simpan mapping antara internal_id (int64) <-> doc_id (string)
    faiss_ids = []
    vectors = []
    bulk_ops = []

    for internal_id, doc in enumerate(docs):
        vec = get_embedding(doc["content"])
        vectors.append(vec)
        faiss_ids.append(internal_id)

        # Simpan ke Elasticsearch
        bulk_ops.append({
            "_index": ES_INDEX,
            "_id": doc["id"],
            "_source": {
                "title": doc["title"],
                "content": doc["content"],
                "embedding": vec.tolist()
            }
        })

    # Hantar bulk ke ES
    helpers.bulk(es, bulk_ops)

    # Convert ke numpy dan masukkan ke FAISS
    vectors_np = np.vstack(vectors).astype("float32")
    faiss_ids_np = np.array(faiss_ids, dtype="int64")

    # Kalau nak ID terikat: kita guna IndexIDMap
    index_with_ids = faiss.IndexIDMap(index)
    index_with_ids.add_with_ids(vectors_np, faiss_ids_np)

    # Simpan juga mapping doc_id
    id_map = {internal_id: doc["id"] for internal_id, doc in enumerate(docs)}

    return index_with_ids, id_map


# =========================
# 7. SEARCH FUNCTION
# =========================

def hybrid_search(query: str, faiss_index, id_map, top_k: int = TOP_K):
    """
    Flow:
    1) Embed query
    2) Search di FAISS → dapat internal_ids + distances
    3) Translate internal_ids ke doc_id (string) guna id_map
    4) mget dokumen dari Elasticsearch
    """

    # 1) Embed query
    q_vec = get_embedding(query).astype("float32")[None, :]  # shape (1, dim)

    # 2) Search FAISS
    distances, internal_ids = faiss_index.search(q_vec, top_k)

    # 3) Map ke doc_id
    internal_ids_list = internal_ids[0]
    doc_ids = [id_map[i] for i in internal_ids_list if i in id_map]

    # 4) Ambil dokumen daripada ES
    if not doc_ids:
        return []

    resp = es.mget(index=ES_INDEX, ids=doc_ids)

    results = []
    for hit, dist, internal_id in zip(resp["docs"], distances[0], internal_ids_list):
        if not hit["found"]:
            continue
        source = hit["_source"]
        results.append({
            "doc_id": hit["_id"],
            "title": source["title"],
            "content": source["content"],
            "faiss_distance": float(dist),
            "internal_id": int(internal_id),
        })

    return results


# =========================
# 8. MAIN DEMO
# =========================

if __name__ == "__main__":
    check_es_connection()
    # 1) Setup index ES
    create_es_index()

    # 2) Build FAISS index + mapping
    faiss_index, id_map = build_faiss_index(documents)

    # 3) Cuba query
    query = "macam mana guna FAISS untuk vector search?"
    print(f"Query: {query}\n")

    results = hybrid_search(query, faiss_index, id_map, top_k=3)

    for rank, r in enumerate(results, start=1):
        print(f"[{rank}] {r['title']} (id={r['doc_id']}, distance={r['faiss_distance']:.4f})")
        print(f"    {r['content']}\n")
