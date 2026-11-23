# ResearchLens AI — LangChain + Qdrant Hybrid RAG

A domain-specific **research-paper Ask My Docs** API. Supports PDF/DOCX/TXT/MD ingestion, LangChain chunking/embeddings, **persistent Qdrant vector database**, BM25, RRF hybrid fusion, cross-encoder reranking, numbered citations, and CI-gated retrieval evaluation.

## Quickstart (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Set OPENAI_API_KEY in .env for LLM-generated answers.
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`. Upload a file via `/documents/upload`, then query `/ask`:

```json
{"question":"How does rainfall affect mosquito breeding?","mode":"hybrid","top_k":5}
```

Without `OPENAI_API_KEY`, the API returns **extractive passages**, not synthesized generative answers. Model downloads occur on first use. Qdrant persists to `data/qdrant` and a JSONL BM25 corpus persists to `data/corpus.jsonl`.

## Evaluation strategy

1. Index a fixed, licensed collection of ~20–30 domain papers; record hashes and model versions.
2. Create `evaluation/gold.json` with **at least 20 human-verified questions** (ideally 200), each with `question` and `relevant_chunk_ids`. Use chunk IDs from `data/corpus.jsonl`. Include factual, semantic, multi-hop, comparison, numerical, and unanswerable questions. Avoid tuning on the final test set.
3. Run `python -m evaluation.run` to compare vector-only, BM25+vector, and hybrid+reranking on **Recall@10, Precision@10, MRR@10, nDCG@10**. Then run `python -m evaluation.gate` to enforce thresholds. The gate fails if data/results are missing or fewer than 20 questions are provided.
4. Run `pytest -q` for fast deterministic tests. GitHub Actions runs both unit tests and the benchmark; **the benchmark job will fail until you commit/provision a reproducible evaluation corpus and labeled questions**. Configure branch protection to require both jobs only after benchmark provisioning.
5. **Generation evaluation (recommended extension):** build a second labeled dataset with reference answers and claim-to-source annotations; evaluate answer correctness, grounded claim support, abstention precision/recall, citation coverage, and hallucination rate. Use human review and/or Ragas/LLM-as-judge, record judge model/version, and gate releases on a fixed held-out set. Measure p50/p95 latency, cost, and regressions separately. These advanced metrics are **not implemented** in the current code.

## Citation security limitation

The code checks that the LLM's citation numbers resolve to supplied passages and refuses invalid/missing citations. It **does not prove entailment** between each claim and cited evidence. Add sentence-level claim decomposition + NLI/human-reviewed verification before calling the application fully citation-enforced. Source excerpts are untrusted text; the prompt explicitly instructs the model to ignore document-embedded instructions, but prompt-injection risk remains.

## Deployment caveats

This is a reference implementation, **not hardened multi-tenant production infrastructure**. Local embedded Qdrant should run as a single process. For production use Qdrant server/cloud, a persistent BM25 service/index, atomic ingestion, user authentication, per-document access control, file scanning, request limits, secrets management, retries, observability, and robust PDF table/OCR handling. Do not upload confidential data without access controls.
