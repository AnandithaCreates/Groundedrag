# GroundedRAG

A RAG assistant with one job done properly: **never answer with something it can't point to**.

Most basic RAG demos retrieve some chunks and generate an answer, full stop. The failure mode nobody's tutorial handles well is the LLM quietly filling gaps with plausible-sounding text that isn't actually in the source documents. GroundedRAG's entire design is built around catching that:

1. **Retrieve** — embed the query, pull the top-k chunks from Qdrant.
2. **Generate** — answer using only the retrieved chunks, via Portkey → Groq, citing the chunk ID for every claim.
3. **Critique** — a second LLM call checks: does every citation in the answer actually point to a chunk that supports the claim next to it? If not, the answer is rejected and regenerated (capped at 2 tries).
4. **Refuse** — if retrieval comes back with nothing relevant, the agent says so instead of guessing.

## Stack, and why each piece is there

| Piece | Choice | Why this one earns its place |
|---|---|---|
| Orchestration | LangGraph (cyclic graph) | Explicit generate → critique → retry loop, the centerpiece of the project |
| Vector DB | **Qdrant Cloud** (free tier) | Real managed vector DB — HNSW approximate search, not a brute-force flat file. The thing to know cold: FAISS-style flat search is exact but linear; Qdrant/HNSW is approximate but scales sub-linearly, which is why a managed vector DB earns its place once a corpus grows |
| LLM Gateway | **Portkey** (free Developer tier) | Every LLM call routes through Portkey instead of hitting Groq directly — gives a single choke point for retries/fallback config, plus gateway-level caching (see below) |
| Cache | **Portkey's built-in "simple" cache** | Exact-match caching on identical prompts, included free. Semantic (similarity-based) caching is a Portkey paid-tier feature — a documented cost tradeoff, not an oversight |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) | Zero cost, runs in the container, no external API |
| Evals | **RAGAS** (`evals/run_evals_ragas.py`) + hand-coded metrics (`evals/run_evals.py`) | RAGAS for the industry-standard library; hand-coded versions kept alongside so you can explain the exact formula underneath, and as a fallback since RAGAS has a genuinely fragile dependency chain |
| Guardrail | Hand-written Portkey/Groq prompt checks | Deliberately NOT a framework (no NeMo Guardrails) — short enough to read end-to-end in an interview, and "I wrote this myself" is a stronger answer here than "I imported a framework" |
| Compute | Cloud Run (1 service) | Free tier covers demo-level traffic |
| IaC | Terraform | Cloud Run service + IAM, minimal footprint (no GCS bucket needed — Qdrant hosts the index) |

## Project layout

```
app/
  config.py        # env vars, model names
  vector_store.py  # Qdrant wrapper: build/upsert, search
  llm_client.py     # Portkey-wrapped chat completion (gateway + cache)
  guardrails.py     # pre-check (on-topic) + post-check (citations are real)
  graph.py          # LangGraph agent: understand -> retrieve -> generate -> critique -> loop/end
  main.py           # FastAPI app, serves the UI + /query
  ingestion.py      # chunking logic
data/sample_docs/   # replace with YOUR documents
scripts/build_index.py   # run this after adding your docs
evals/
  golden_set.json       # hand-written Q&A pairs against your docs
  metrics.py             # faithfulness / relevancy / retrieval precision, hand-coded
  run_evals.py            # runs golden_set.json through the hand-coded metrics
  run_evals_ragas.py      # runs the same set through RAGAS's standard metrics
static/index.html   # minimal chat UI
terraform/           # Cloud Run + IAM
```

## Setting up the free-tier accounts (one-time, ~10 min)

1. **Groq** — free key at console.groq.com
2. **Portkey** — free account at app.portkey.ai → Virtual Keys → Add → paste your Groq key → copy the resulting virtual key ID
3. **Qdrant Cloud** — free cluster at cloud.qdrant.io → copy the cluster URL and API key
   - Free clusters auto-suspend after 1 week idle, delete after 4 weeks. Reactivate from the console before a demo if it's been quiet.

## Running it locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # fill in Groq / Portkey / Qdrant credentials

# 1. drop your own .md/.txt files into data/sample_docs/ (sample files included)
# 2. build the index in Qdrant
python scripts/build_index.py

# 3. run the API + UI
uvicorn app.main:app --reload --port 8000
# open http://localhost:8000
```

## Running the evals

```bash
python evals/run_evals.py          # hand-coded metrics, zero fragile deps
python evals/run_evals_ragas.py    # RAGAS's standard metrics (needs the eval extras installed)
```

Both print per-question scores and save a `results*.json` file. Expect the numbers to roughly track each other — if they diverge a lot, that's worth digging into (and a good interview story either way).

## Deploying (free tier)

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT/groundedrag

cd terraform
terraform init
terraform apply \
  -var="project_id=YOUR_PROJECT" \
  -var="container_image=gcr.io/YOUR_PROJECT/groundedrag" \
  -var="portkey_api_key=YOUR_PORTKEY_KEY" \
  -var="portkey_virtual_key=YOUR_PORTKEY_VIRTUAL_KEY" \
  -var="qdrant_url=YOUR_QDRANT_URL" \
  -var="qdrant_api_key=YOUR_QDRANT_KEY"
```

One Cloud Run service, capped at low CPU/memory and max 2 instances — comfortably inside the always-free tier at demo traffic. Run `terraform destroy` when you're done recording your demo.

## What's genuinely a tradeoff here (say this in an interview, don't hide it)

- Portkey's free tier only includes exact-match caching, not semantic caching — a repeated *identical* question is fast, a *rephrased* one isn't. Semantic caching is the real upgrade path, gated behind Portkey's paid tier.
- Qdrant's free cluster is single-node with no HA — fine for a demo, not for production traffic.
- The critique loop adds latency (1 extra LLM call per attempt) — capped at 2 retries specifically to bound cost/latency, a deliberate tradeoff, not an accident.
- RAGAS's dependency chain is fragile enough that it's worth having the hand-coded fallback — a real resilience decision, not padding.
