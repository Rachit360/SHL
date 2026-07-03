# SHL Assessment Advisor

A conversational AI agent that helps hiring managers and recruiters find the right SHL assessments through dialogue — turning a vague hiring intent into a grounded, catalog-backed shortlist.

**Live API:** [shl-assessment-advisor-bpcp.onrender.com](https://shl-assessment-advisor-bpcp.onrender.com) → redirects to interactive Swagger docs (`/docs`)

Built for the SHL Research Intern take-home assignment.

---

## Problem

Recruiters often don't know the right vocabulary until they describe the role out loud. Keyword-based catalog search assumes they already do. This agent instead takes a vague intent (*"I am hiring a Java developer"*) and guides the user to a grounded shortlist through conversation — clarifying only when genuinely necessary, never guessing.

## API

The service is **fully stateless** — every request carries the complete conversation history.

### `POST /chat`
```json
{
  "messages": [
    {"role": "user", "content": "I am hiring a Java developer"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```
```json
{
  "reply": "Here are assessments for a mid-level Java developer.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

### `GET /health`
```json
{"status": "ok"}
```

---

## Architecture

```
User
  │
  ▼
FastAPI POST /chat
  │
  ▼
Deterministic Scope Guard  ── blocks prompt injection & off-topic requests before any LLM call
  │
  ▼
Conversation State Router  ── CLARIFY · RECOMMEND · REFINE · COMPARE · REFUSE
  │
  ▼
Constraint Extraction (Groq LLM)
  │
  ▼
ChromaDB Semantic Retrieval + Multi-Factor Ranking
  │
  ▼
Groq LLaMA-3.3-70B + Catalog Context
  │
  ▼
Validation · Dedup · Cap → Response
```

### Conversation states

| State | Trigger | `recommendations` |
|---|---|---|
| **CLARIFY** | Job role/context unknown | `[]` |
| **RECOMMEND** | Enough context to act | 1–10 items |
| **REFINE** | User changes constraints mid-conversation | Updated list |
| **COMPARE** | User asks to compare named assessments | `[]` (grounded reply only) |
| **REFUSE** | Off-topic, legal, salary, prompt injection | `[]` |

### Deterministic pre-routing

Several behaviors are handled by regex-based gates that run **before** the LLM, rather than relying purely on prompt instructions — this makes them predictable and independently testable:

- **Scope guard** — blocks prompt injection and off-topic requests (salary, legal advice, certifications, courses) before any LLM call.
- **Executive/leadership roles** — asks for hiring purpose (selection, development, succession planning) before recommending.
- **Unavailable technologies** (e.g. Rust) — informs the user the catalog has no dedicated assessment and asks before substituting alternatives.
- **Contact centre roles** — confirms language before recommending call-handling assessments.
- **Healthcare admin + multilingual requirements** — flags the catalog's mixed language availability (e.g. HIPAA/Medical Terminology are English-only) and asks for clarification.
- **Mixed backend/frontend job descriptions** — asks whether the role is backend-, frontend-, or full-stack-focused before building a battery.
- **User confirmation detection** — recognizes acceptance phrases ("that works", "we'll go with that") and sets `end_of_conversation: true` with the finalized shortlist.
- **Pinned exact-match batteries** — a small set of well-defined scenarios (e.g. graduate management trainee batteries, quick Excel/Word screens) map to specific, exact-matched catalog assessments rather than open-ended retrieval, with no substitution risk.

---

## Data Pipeline

The official SHL catalog contains 377 entries, including reports and candidate packs that must never be recommended.

1. **Scrape** — Playwright extracts Individual Test Solutions from `online.shl.com` via direct DataTable JS execution.
2. **Filter** — 377 → 317 valid assessments, removing reports and profile cards.
3. **Merge** — the official catalog provides ground-truth URLs and durations; scraped data adds semantic enrichment.
4. **Enrich** — each entry gains `skills_measured`, `keywords`, `recommend_for`, `business_problems_solved`, and a rich `embedding_text`.
5. **Embed** — `BAAI/bge-small-en-v1.5` generates 384-dim normalized vectors, stored in ChromaDB with cosine similarity.

## Retrieval & Ranking

Semantic retrieval alone under-recommended behavioral/personality assessments for technically-worded queries. This is addressed with:

- A **behavioral signal detector** that runs a parallel retrieval pass for personality/interpersonal terms and merges it with the primary technical pool.
- A **weighted multi-factor ranking layer** (`rag/ranking.py`) scoring each candidate across 9 independent signals — technical skill match, job role relevance, seniority match, behavioral match, requested test type, adaptive requirement, remote requirement, language requirement, and semantic similarity — then sorting descending.
- **Exact-name matching** (`rag/catalog_lookup.py`) for the COMPARE state, avoiding semantic-similarity mixups between similarly named assessments (e.g. "Verify Numerical" vs "Verify Numerical Reasoning"). If a name is ambiguous or not found, the agent asks for clarification rather than guessing.

Every recommendation is validated against the retrieval pool by exact name/URL match before being returned — anything not verbatim present in the catalog is dropped. URLs not originating from `shl.com` are never returned.

---

## Tech Stack

| Component | Choice |
|---|---|
| API framework | FastAPI + Uvicorn |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Embeddings | `BAAI/bge-small-en-v1.5` (SentenceTransformers) |
| Vector store | ChromaDB |
| Deployment | Render.com |
| Python | 3.11.9 (pinned) |

---

## Project Structure

```
├── api/
│   ├── main.py           # FastAPI app, startup, / → /docs redirect
│   ├── routes.py          # /health, /chat, state routing, all deterministic gates
│   └── schemas.py          # Pydantic request/response models
├── rag/
│   ├── embedder.py         # builds the ChromaDB index (offline, one-time)
│   ├── retriever.py         # BGE query encoding + ChromaDB search
│   ├── vector_store.py       # ChromaDB wrapper
│   ├── ranking.py          # weighted multi-factor scoring
│   ├── catalog_lookup.py     # exact-name matching for comparisons/pinned batteries
│   └── prompts.py          # system prompts, catalog formatting
├── scraper/              # offline catalog scraping/cleaning (not used at runtime)
├── data/
│   ├── assessments.json      # final 317-assessment catalog (ground truth)
│   └── chroma_db/          # persisted vector index
├── requirements.txt
└── runtime.txt            # pins python-3.11.9
```

---

## Running Locally

```bash
pip install -r requirements.txt
# create a .env file with:
# GROQ_API_KEY=your_key_here
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Then visit `http://127.0.0.1:8000/docs` for the interactive API explorer, or:
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I am hiring a Java developer"}]}'
```

---

## Notable Trade-offs

| Decision | Reasoning |
|---|---|
| BGE-small over larger embedding models | Fits comfortably in Render's free-tier 512MB RAM; keeps latency under the 30s evaluator timeout |
| Deterministic regex gates over pure LLM routing | Predictable, unit-testable, and immune to LLM non-determinism for known edge cases |
| Stateless API | Simpler to scale and reason about; trade-off is sending full history on every request |
| Groq free tier | `llama-3.3-70b` is capable enough for structured JSON generation at zero infrastructure cost |

## What Didn't Work Initially

- Semantic retrieval alone missed behavioral assessments for technically-worded queries → fixed with ranking + a dedicated behavioral retrieval pass.
- REFINE occasionally returned an empty shortlist when the LLM couldn't reconcile constraints → fixed with a finalization fallback that backfills from the ranked retrieval pool.
- Render deployment initially failed on Python 3.14 (no prebuilt wheel for `pydantic-core`) → fixed by pinning `PYTHON_VERSION=3.11.9`.
- A ChromaDB version mismatch between the local build environment and Render caused a startup crash → fixed by pinning the exact local version in `requirements.txt`.

---

## License

MIT — see [LICENSE](LICENSE).
