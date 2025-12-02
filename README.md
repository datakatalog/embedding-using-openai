# Embedding Using OpenAI

Hands-on scripts for generating OpenAI text embeddings, visualising them, and using them inside a simple Chroma vector database (with a small Netflix-style dataset).

---

## Repository Map
- `01-embedding_intro.py` - create a single embedding for a sentence and print token usage.
- `02-tsne.py` - embed sample news headlines, project them to 2D with t-SNE, and run a cosine-similarity search for a query term.
- `03-vector_database.py` - build a persistent Chroma collection, add sample docs plus the Netflix dataset from `netflix_data.py`, run semantic search, and estimate embedding cost.
- `netflix_data.py` - curated titles, descriptions, categories, and release years used to seed Chroma.

---

## Requirements
- Python 3.10+
- An active OpenAI API key in a `.env` file (`OPENAI_API_KEY=...`)
- Install dependencies: `pip install -r requirements.txt`

### Optional: set up a virtual environment (Windows PowerShell)
```pwsh
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## How to Run the Examples

### 1) Single embedding
```pwsh
python 01-embedding_intro.py
```
- Verifies `OPENAI_API_KEY` is available and prints the token count for one example sentence.

### 2) Headline clustering + search
```pwsh
python 02-tsne.py
```
- Embeds 10 example news headlines, reports embedding lengths, and runs a cosine-similarity search for the word `computer`.
- t-SNE reduces embeddings to 2D; to see the scatter plot, uncomment `plt.show()` near the bottom.

### 3) Vector database with Chroma + Netflix mini-set
```pwsh
python 03-vector_database.py
```
- Creates a persistent Chroma store in `./chroma_db`, seeds sample docs and the Netflix dataset, then performs semantic search (example query: movies with singing/dancing).
- Prints counts, peeked docs, retrieved items by id, search results with distances, and a rough token/cost estimation using `tiktoken`.
- Re-running keeps the persisted data; delete `chroma_db/` if you want a clean start.


### `04-hybrid_search.py`
A **hybrid semantic search pipeline** combining:

- OpenAI Embeddings (`text-embedding-3-small`)
- **FAISS** → ANN engine (fast vector search)
- **Elasticsearch** → metadata + storage + full-text search

Flow:
1. Load API key from `.env`
2. Connect to local Elasticsearch (`http://localhost:9200`)
3. Create ES index + insert documents
4. Generate embeddings
5. Insert vectors into FAISS using `IndexIDMap`
6. Run hybrid ANN search and display top-K results

**Example Output:**
```pwsh
[OK] Connected to Elasticsearch cluster: docker-cluster
Query: macam mana guna FAISS untuk vector search?

[1] FAISS untuk ANN (id=doc-3, distance=0.72)
FAISS digunakan untuk approximate nearest neighbour search.

[2] Scaling Vector Search (id=doc-5, distance=0.88)
Teknik scaling, sharding dan replication untuk vector search.

[3] Pengenalan Vector Database (id=doc-1, distance=0.98)
Ini penerangan ringkas tentang vector DB.
```
Running Elasticsearch (Required for 04-hybrid_search.py)

✔ Recommended: Run Elasticsearch via Docker
```pwsh
docker run --name es-dev -p 9200:9200 ^
  -e "discovery.type=single-node" ^
  -e "xpack.security.enabled=false" ^
  docker.elastic.co/elasticsearch/elasticsearch:8.15.0
```

PowerShell backtick version:
```pwsh
docker run --name es-dev `
  -p 9200:9200 `
  -e "discovery.type=single-node" `
  -e "xpack.security.enabled=false" `
  docker.elastic.co/elasticsearch/elasticsearch:8.15.0
```
After first run:
```pwsh
docker start es-dev
docker stop es-dev
```
---

## Notes
- The OpenAI embedding model used is `text-embedding-3-small` (adjust in the scripts if you prefer another model).
- API usage incurs cost; the provided estimation in `03-vector_database.py` uses a simple USD-to-MYR conversion you can tweak.
- Feel free to replace the sample text lists or Netflix entries with your own data to try different search behaviours.
- You may replace sample documents in 04-hybrid_search.py with PDFs, guidelines, policies, etc.

- The pipeline is RAG-ready:
just add a function that sends FAISS+ES results to an LLM to generate natural-language answers.
