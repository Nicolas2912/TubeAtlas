# Implementation Plan for Subtask 5.4 – Triple Merging & Deduplication Logic

*(Task reference: Build Knowledge Graph Generation Pipeline – Triple Merging & Deduplication Logic)*

## 1  Overview
The goal of this sub-task is to design a **deterministic, idempotent, and scalable** mechanism that incrementally merges freshly-extracted triples into the persistent knowledge graph (KG) while:

* Preventing _exact_ and _near-duplicate_ triples from bloating the graph.
* Preserving full provenance (chunk ID, transcript ID, LLM model, timestamp, cost)
* Resolving conflicting attribute values via a transparent heuristic (recency ⚖ confidence).
* Continuously updating auxiliary edge attributes (frequency, sources, mean confidence).

The component will live in `src/tubeatlas/services/kg/triple_merger.py` and will be consumed by `KGService.ingest_triples()`.

> **Key design principles**: Functional core + typed dataclasses, pure-Python ❤️ with NetworkX for graph ops, fast batch processing, and easy unit-testing.

---

## 2  Data Structures
```python
@dataclass(slots=True)
class Triple:
    subject: str
    predicate: str
    object: str
    confidence: float  # 0.0-1.0 from GraphPrompter
    provenance: Provenance  # chunk_id, video_id, channel_id, llm_model, t_created
```

```python
@dataclass(slots=True)
class Provenance:
    chunk_id: str
    video_id: str
    channel_id: str
    llm_model: str
    t_created: datetime
```

```python
@dataclass(slots=True)
class MergeStats:
    inserted: int
    exact_dupes: int
    fuzzy_dupes: int
    updated: int  # edge attr refreshes
```

The **canonical key (CK)** for a triple is a SHA-256 hash of a normalised string:
```
f"{norm(s)}|{norm(p)}|{norm(o)}"  → sha256.hexdigest()[:16]
```
where `norm()` lower-cases, trims, collapses whitespace, and strips punctuation for high recall.

---

## 3  Algorithm
1. **Preparation**
   1. Load the current KG (`networkx.MultiDiGraph`) and a locally-cached **signature index** (`dict[CK, edge_id]`). The index is persisted alongside the KG JSON for O(1) re-load.
   2. Warm a FAISS `IndexFlatIP` embedding index containing `(embedding32, CK)` pairs for fuzzy look-ups.

2. **Batch Merge (`merge_batch(triples: list[Triple])`)**
   ```text
   for t in triples:
       ck = canonical_key(t)
       if ck in sig_idx:                # 3.1 Exact duplicate
           update_edge_attrs(edge, t)
           stats.exact_dupes += 1
           continue

       nn_cks = query_faiss(embedding(t), k=5, min_sim=SIM_THR)  # 3.2 Fuzzy dupes
       if nn_cks:
           edge = select_best_edge(nn_cks)
           update_edge_attrs(edge, t, fuzzy=True)
           stats.fuzzy_dupes += 1
           continue

       # 3.3 Brand-new triple – add edge
       add_edge(graph, t, ck)
       stats.inserted += 1
   ```

3. **Attribute Refresh (`update_edge_attrs`)**
   * `frequency += 1`
   * `confidence = harmonic_mean(confidence, t.confidence)`
   * Append `provenance` dict to `edge["sources"]` (capped at last 10 for storage limits)
   * `edge["last_seen"] = now()`

4. **Conflict Resolution**
   * If subject/predicate/object text differs _semantically_ (fuzzy path) choose winner via:
     * Higher **confidence** wins; break ties with **recency**.
   * Keep looser variant as `alias` attribute for downstream NLP.

5. **Persistence**
   * On successful batch merge, dump KG via Persistence Layer (subtask 5.5) & flush `sig_idx` + FAISS index.

6. **Complexity**
   * Exact check      O(N)
   * Fuzzy check      O(N · log M) with FAISS (≈ constant) → linear in practise.

---

## 4  Modules & Functions
| Module | Responsibility |
|--------|----------------|
| `triple_merger.py` | public `TripleMerger` class + helpers |
| `embeddings.py` | thin wrapper around `OpenAIEmbedder` returning 768-d np.array |
| `signature.py` | canonicalisation & SHA-256 hashing |
| `index.py` | load/save signature dictionary & FAISS index |

Key API:
```python
class TripleMerger:
    def __init__(self, kg_path: Path, sim_thr: float = 0.92): ...
    def merge_batch(self, triples: Sequence[Triple]) -> MergeStats: ...
```

---

## 5  Performance & Memory Considerations
* Keep signature dict in memory; 1 M triples ≈ 80 MB (16-byte key + 48-byte ptr).
* FAISS index uses **IndexBinaryFlat** (16-byte signatures) for 8× memory cut.
* Batch size defaults to 1 000; large uploads stream in chunks to bound RAM.

---

## 6  Logging & Observability
* Use `structlog` with JSON renderer.
* Log per-batch summary (stats + duration + mem usage) at **INFO**.
* Emit `triple_merge_conflict` events at **DEBUG** with edge ids.

---

## 7  Testing Strategy
1. **Unit**
   * `test_signature_uniqueness` – CK uniqueness across permutations.
   * `test_exact_dupe_merge` – same triple twice ⇒ edge frequency == 2.
   * `test_fuzzy_dupe_merge` – "YouTube" vs "Youtube" ⇒ dedup.
   * `test_conflict_resolution` – differing confidence/recency.
2. **Property-based** (`hypothesis`) generate random triples; verify idempotency.
3. **Benchmark** (pytest-benchmark) ensure ≤ 50 ms to merge 1 000 triples.

---

## 8  Integration Steps
1. Implement modules & tests (TDD).
2. Wire into `KGService.ingest_triples()` replacing current stub.
3. Update Celery `generate_video_kg` task to call merger after extraction.
4. Add migration script to build initial signature + FAISS index from existing KG files.

---

## 9  Done Criteria
* All unit/property/benchmark tests pass (coverage ≥ 90%).
* CI pipeline green; pre-commit lint/type checks clean.
* Merging a sample channel twice yields identical KG JSON (proving idempotency).
* Performance benchmark recorded <50 ms / 1 000 triples on M1 Pro.

---

**ETA**: 6 pomodoros (≈ 3 hrs) for coding & tests, 1 pomodoro for docs.