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
    print(f"{i+1}. {meta['title']}  (distance={dist:.4f})")