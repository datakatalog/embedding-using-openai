import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from netflix_data import netflix_dataset

# =========================
# LOAD API KEY
# =========================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise Exception("Please set OPENAI_API_KEY in .env file")

# =========================
# CONNECT TO CHROMA (PERSIST MODE)
# =========================
client = chromadb.PersistentClient(path="./chroma_db")  # folder akan dibuat automatik

print("Connected to ChromaDB.")

# =========================
# CREATE COLLECTION
# =========================
embedding_fn = OpenAIEmbeddingFunction(
    model_name="text-embedding-3-small",
    api_key=api_key
)

collection = client.get_or_create_collection(
    name="my_collection",
    embedding_function=embedding_fn
)

print("Created collection:", collection.name)

# =========================
# ADD DOCUMENTS
# =========================
docs = [
    "This is the source text",
    "This is document 1",
    "This is document 2"
]

ids = ["my-doc", "my-doc-1", "my-doc-2"]

collection.add(
    ids=ids,
    documents=docs
)

print("Documents added.")

# =========================
# COUNT DOCUMENTS
# =========================
print("Total docs in collection:", collection.count())

# =========================
# INSPECT COLLECTION
# =========================
peek = collection.peek()
print("\nPEEK RESULT:")
print(peek)

# =========================
# RETRIEVE A SPECIFIC ITEM (BY ID)
# =========================
item = collection.get(ids=["my-doc"])
print("\nRETRIEVED ITEM:")
print(item)

# =========================
# SEMANTIC SEARCH (QUERY)
# =========================
query_text = "source material"
results = collection.query(
    query_texts=[query_text],
    n_results=2
)

print("\nSEMANTIC SEARCH RESULT:")
print(results)

# =========================
# ESTIMATE COST OF EMBEDDING
# =========================
import tiktoken

enc = tiktoken.encoding_for_model("text-embedding-3-small")

total_tokens = sum(len(enc.encode(x)) for x in docs)
cost_per_1k = 0.00002
cost = cost_per_1k * (total_tokens / 1000)

print("\nTOKEN & COST ESTIMATION:")
print("Total tokens:", total_tokens)
print("Estimated Cost: $", cost)

usd_rate = 4.70  # 1 USD ≈ RM4.70 (bro boleh adjust)

# Cost per token for text-embedding-3-small
cost_per_million = 0.02
cost_usd = (total_tokens / 1_000_000) * cost_per_million

# Convert to MYR
cost_myr = cost_usd * usd_rate

print("TOKEN & COST ESTIMATION:")
print(f"Total tokens: {total_tokens}")
print(f"Estimated Cost (USD): ${cost_usd:.10f}")
print(f"Estimated Cost (MYR): RM{cost_myr:.10f}")


# =========================
# NETFLIX SECTION
# =========================

def build_netflix_text(item: dict) -> str:
    """Gabungkan title + description + categories + type + year."""
    cats = ", ".join(item["categories"])
    return (
        f"Title: {item['title']}\n"
        f"Description: {item['description']}\n"
        f"Categories: {cats}\n"
        f"Type: {item['type']}\n"
        f"Year: {item['release_year']}"
    )

client.delete_collection("netflix_collection")

netflix_collection = client.get_or_create_collection(
    name="netflix_collection",
    embedding_function=embedding_fn,
)

netflix_ids = [item["id"] for item in netflix_dataset]
netflix_docs = [build_netflix_text(item) for item in netflix_dataset]
netflix_metadatas = [
    {
        "title": item["title"],
        "type": item["type"],
        "year": item["release_year"],
        # jadikan categories sebagai satu string, bukan list
        "categories": ", ".join(item["categories"]),
    }
    for item in netflix_dataset
]

if netflix_collection.count() == 0:
    netflix_collection.add(
        ids=netflix_ids,
        documents=netflix_docs,
        metadatas=netflix_metadatas,
    )
    print("\nInserted Netflix docs:", netflix_collection.count())
else:
    print("\nNetflix docs already exist:", netflix_collection.count())

# =========================
# EXAMPLE SEMANTIC SEARCH
# =========================
query = "movies where people sing and dance a lot"
result = netflix_collection.query(
    query_texts=[query],
    n_results=3,
)

print("\n=== NETFLIX RECOMMENDATION RESULT ===")
for i, doc in enumerate(result["documents"][0]):
    meta = result["metadatas"][0][i]
    dist = result["distances"][0][i]
    # fallback kalau metadata tiada / None
    if isinstance(meta, dict) and "title" in meta:
        title = meta["title"]
    else:
        # ambil line pertama dari dokumen sebagai title sementara
        title = doc.split("\n")[0]
    print(f"{i+1}. {meta['title']}  (distance={dist:.4f})")


# =========================
# EXAMPLE: UPDATE DOCUMENT
# =========================
print("\n--- UPDATE EXAMPLE ---")

netflix_collection.update(
    ids=["s6"],  # id dokumen yang nak diubah
    documents=[
        "Title: La La Land (Movie)\n"
        "Description: An updated description about a jazz pianist and an actress "
        "chasing their dreams in Los Angeles while dealing with love and sacrifice.\n"
        "Categories: Musicals, Romantic Movies\n"
        "Type: Movie\n"
        "Year: 2016"
    ],
)

updated = netflix_collection.get(ids=["s6"])
print("Updated doc for s6:")
print(updated["documents"][0][0])

# =========================
# EXAMPLE: UPSERT DOCUMENTS
# =========================
print("\n--- UPSERT EXAMPLE ---")

netflix_collection.upsert(
    ids=["s1", "s7"],
    documents=[
        # s1: Kota Factory (updated text)
        "Title: Kota Factory (TV Show)\n"
        "Description: Updated text – a black-and-white series about students in a "
        "coaching centre preparing for India's toughest engineering exams.\n"
        "Categories: International TV Shows, TV Dramas\n"
        "Type: TV Show\n"
        "Year: 2019",

        # s7: new show/movie – contoh baru
        "Title: Sing (Movie)\n"
        "Description: A group of animals compete in a singing contest to save a theatre.\n"
        "Categories: Kids' Movies, Musicals\n"
        "Type: Movie\n"
        "Year: 2016",
    ],
)

print("Total docs after upsert:", netflix_collection.count())

# =========================
# EXAMPLE: DELETE BY IDS
# =========================
print("\n--- DELETE EXAMPLE ---")

before = netflix_collection.count()
netflix_collection.delete(ids=["s5"])
after = netflix_collection.count()

print(f"Deleted id 's5'. Count before: {before}, after: {after}")

print("\nPEEK RESULT:")
print(peek)

# =========================
# DANGER ZONE: RESET DB
# =========================
# WARNING: ini akan buang SEMUA collection & dokumen dalam ChromaDB!
# Uncomment kalau betul-betul nak reset.
#
# print("\n!!! RESETTING ENTIRE CHROMA DB !!!")
# client.reset()
# print("All collections & items deleted.")
