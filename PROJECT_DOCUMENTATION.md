# Project Documentation — VL-JEPA vs CLIP for CCTV Event Retrieval

> Complete architecture walkthrough and file-by-file reference

---

## 1. Project Overview

This capstone project compares **CLIP** and **VL-JEPA** vision-language models for retrieving relevant CCTV footage using natural-language queries. It investigates three research hypotheses:

| # | Hypothesis | Core Question |
|---|---|---|
| H1 | Retrieval Quality | Does VL-JEPA match or exceed CLIP for text→video retrieval? |
| H2 | Cross-Camera Generalization | Do embeddings generalize across aerial / ground / rooftop cameras? |
| H3 | Faithful Report Generation | Does grounding LLM reports in evidence reduce hallucination? |

**Datasets used**: UCF-Crime, UCF-ARG, RLVS (Real Life Violence Situations), TinyVIRAT-v2

---

## 2. Architecture Diagram

```
 ┌─────────────┐     ┌────────────────┐     ┌──────────────┐     ┌─────────────┐
 │  Raw CCTV   │────▶│ Dataset Loader │────▶│   Encoder    │────▶│ Embeddings  │
 │  Videos     │     │   (data/)      │     │ CLIP|VL-JEPA │     │ .npy + .json│
 └─────────────┘     └────────────────┘     └──────────────┘     └──────┬──────┘
                                                                        │
                                                              ┌─────────▼────────┐
                                                              │  FAISS Index     │
                                                              │  (.faiss file)   │
                                                              └─────────┬────────┘
                          ┌─────────────────────────────────────────────┤
                          │                    │                        │
                ┌─────────▼──────┐   ┌────────▼─────────┐   ┌─────────▼──────────┐
                │ Text Retrieval │   │  LLM Reporter    │   │   Evaluation       │
                │ (top-K clips)  │   │  (incident report│   │ (Recall, MRR, mAP, │
                └────────────────┘   │  + grounding)    │   │  latency, faithful)│
                                     └──────────────────┘   └────────────────────┘
```

**Flow**: Videos → Frame sampling → Encoding → FAISS indexing → Query retrieval → Report generation → Evaluation

---

## 3. Repository Structure

```
Capstone-Project/
│
├── reference/                      Research documents & presentations
│   ├── CCTV_VL-JEPA_Capstone_Research_Report (1).pdf
│   ├── Literature_Survey_Report.pdf
│   ├── Literature_Survey_report_VL-JEPA_vs_CLIP_for_CCTV_Event_Retrieval.pdf
│   ├── VL-JEPA_vs_CLIP_for_CCTV_Event_Retrieval.pptx
│   ├── DATASET_CARDS.md
│   ├── project ppt content text.txt
│   ├── kilocode_agent_prompt.md
│   ├── technicalrequirements.png
│   └── workflow.png
│
└── implementation/                 All source code
    ├── config/config.yaml          Master configuration
    ├── data/                       Dataset loaders (4 benchmarks)
    ├── models/                     CLIP & VL-JEPA encoders
    ├── retrieval/                  FAISS indexing & search
    ├── reporting/                  LLM report generation
    ├── evaluation/                 Metrics & benchmarks
    ├── scripts/                    CLI entry points (pipeline)
    ├── demo/                       Streamlit web UI
    ├── tests/                      Unit tests (no data needed)
    ├── utils/                      Shared helpers
    ├── Dockerfile                  Container image
    ├── docker-compose.yml          Service orchestration
    ├── Makefile                    Developer shortcuts
    ├── requirements.txt            Python dependencies
    ├── setup.py                    Package definition
    ├── .env.example                Environment template
    └── .gitignore                  Git exclusions
```

---

## 4. File-by-File Reference

### 4.1 `config/`

| File | Description |
|---|---|
| `config.yaml` | **Central configuration** for the entire project. Contains sections for: **data** (paths and settings per dataset), **video** (frame count=8, resolution=224, decode backend), **models** (CLIP ViT-B/32 primary, V-JEPA 2 ViT-B/16 backbone, alignment strategy), **retrieval** (FAISS index type, top-K=5), **llm** (Flan-T5-base, temperature=0.3, max tokens), **evaluation** (Recall@K values, latency runs, faithfulness thresholds), **experiment** (seed=42, output directory). All paths overridable via `CAPSTONE_DATA_ROOT`, `CAPSTONE_OUTPUT_ROOT`, `CAPSTONE_DEVICE`, `CAPSTONE_CONFIG` env vars. |

---

### 4.2 `data/` — Dataset Loaders

| File | Description |
|---|---|
| `__init__.py` | Exports `DATASETS = ("ucf_crime", "ucf_arg", "rlvs", "tinyvirat")` and `build_dataset()` factory that dispatches to the correct loader class. |
| `base_dataset.py` | Abstract base class `BaseVideoDataset` (extends PyTorch `Dataset`). Handles: reading video config, calling subclass `_build_index()`, decoding frames in `__getitem__()`, and a **corrupted-video retry** mechanism (tries up to 5 neighbors on decode failure). Returns `{"video", "label", "path", "dataset", "metadata"}`. |
| `data_utils.py` | Video I/O utilities: `sample_frames()` — uniform temporal sampling via Decord (preferred) or OpenCV (fallback for Windows). `frames_to_tensor()` — resize + convert to `[T,C,H,W]` float tensor. `normalize_frames()` — channel-wise mean/std normalization. `collate_fn()` — custom DataLoader collation. Also defines `VideoDecodeError`. |
| `ucf_crime_dataset.py` | **UCF-Crime** loader. 13 anomaly classes (Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism). Supports 3 modes: `all` (folder scan), `train` (parse split file), `test` (parse temporal annotations). Skips un-extracted zip parts gracefully. |
| `ucf_arg_dataset.py` | **UCF-ARG** loader. Indexes clips from 3 viewpoints: `aerial_clips`, `ground_clips`, `rooftop_clips`. Parses filenames for actor/run/viewpoint/action metadata. Supports viewpoint filtering for H2 experiments. |
| `rlvs_dataset.py` | **RLVS** loader. Classes: Violence, NonViolence. **Critical**: must glob both `*.mp4` AND `*.avi` — 49 NonViolence clips are AVI and would be silently dropped otherwise. Deduplicates across glob patterns. |
| `tinyvirat_dataset.py` | **TinyVIRAT-v2** loader. Reads JSON split files. Multi-label (each clip can have multiple activities). Preserves native ~72px resolution to avoid premature upsampling. Supports `max_clips` cap for constrained hosts. 26 activity classes. |

---

### 4.3 `models/` — Encoders

| File | Description |
|---|---|
| `__init__.py` | Re-exports `CLIPEncoder`, `VLJEPAEncoder`, `get_encoder`. |
| `clip_encoder.py` | `CLIPEncoder` — wraps HuggingFace `CLIPModel`. `encode_video()`: flattens `[B,T,C,H,W]` to frames, normalizes with CLIP mean/std, extracts image features, mean-pools over time, L2-normalizes. `encode_text()`: tokenizes (max 77 tokens), extracts text features, L2-normalizes. Supports `primary` (ViT-B/32) and `secondary` (ViT-L/14) variants. Uses fp16 on CUDA. |
| `vljepa_encoder.py` | `VLJEPAEncoder` — V-JEPA 2 visual backbone + CLIP text encoder fallback. Loads visual model via `AutoModel`, probes output dim with dummy forward pass. Resamples input to expected frame count. Uses ImageNet normalization. When native VL-JEPA is unavailable, truncates both visual and text embeddings to their shared prefix dimension (this zero-training alignment is itself part of H1). |
| `encoder_factory.py` | `get_encoder(name, config, variant)` — factory function. Clears GPU cache, lazy-imports the correct encoder class, returns initialized encoder. |

---

### 4.4 `retrieval/` — FAISS Indexing & Search

| File | Description |
|---|---|
| `__init__.py` | Re-exports `CCTVRetriever`, `EmbeddingIndexer`, normalization utils. |
| `embedding_indexer.py` | `EmbeddingIndexer` — manages FAISS index. Uses `IndexFlatIP` (cosine similarity for L2-normalized vectors) by default; auto-switches to `IndexIVFFlat` past 20k vectors. Methods: `add()` (validate + insert), `save()` (write `.faiss` + `.meta.json` sidecar), `load()` (restore from disk), `search()` (top-K with scores + metadata). |
| `retriever.py` | `CCTVRetriever` — binds an encoder + indexer. `retrieve(query, k)`: encodes text query → searches FAISS → returns ranked results with scores and metadata. Validates dimension match at construction. |
| `retrieval_utils.py` | Helpers: `l2_normalize_tensor()` (PyTorch), `l2_normalize_np()` (NumPy), `as_float32()` (safe cast for FAISS). |

---

### 4.5 `reporting/` — LLM Incident Reports

| File | Description |
|---|---|
| `__init__.py` | Re-exports `IncidentReporter`, prompt builders. |
| `llm_reporter.py` | `IncidentReporter` — loads HuggingFace LLM (default: `google/flan-t5-base`). Tries Seq2Seq first, falls back to CausalLM. Supports 4-bit/8-bit quantization (Linux). `generate_report(evidence, query, grounded)`: builds prompt → generates text. Grounded mode includes evidence; ungrounded mode (H3 control) uses query only. |
| `prompt_templates.py` | Two prompts: `REPORT_PROMPT` (grounded — with formatted evidence clips, scores, labels) and `UNGROUNDED_PROMPT` (query only). Output structure: Incident Type, Time/Location, Observed Behaviors, Severity Assessment, Recommended Action. Evidence capped at 6000 chars. |
| `evidence_utils.py` | Thin wrapper re-exporting `evidence_to_text()` — stable import target for scripts and demo. |

---

### 4.6 `evaluation/` — Metrics & Benchmarks

| File | Description |
|---|---|
| `__init__.py` | Re-exports all metric functions. |
| `retrieval_metrics.py` | IR metrics: **Recall@K**, **nDCG@K**, **MRR**, **mAP**. `compute_all_metrics()` takes retrieved vs ground-truth label lists, returns a metrics dict. Handles single and multi-label. |
| `efficiency_metrics.py` | Hardware benchmarks: `measure_encoding_latency()` (ms per clip, with CUDA sync), `measure_memory_usage()` (peak GPU MB + CPU RSS), `measure_throughput()` (clips/second). |
| `faithfulness_metrics.py` | H3 evaluation: `grounding_score()` — splits report into sentences, checks token overlap + number/date consistency against evidence. `hallucination_rate()` = 1 − grounding_score. `bertscore_faithfulness()` — optional BERTScore. `compute_faithfulness_report()` — produces a DataFrame with per-report metrics and unsupported-sentence tags. |
| `generalization_eval.py` | H2 evaluation: `cross_viewpoint_matrix()` — pairwise Recall@K matrix across UCF-ARG viewpoints (query from one camera → index from another). `plot_heatmap()` — renders as a viridis-colored PNG. |
| `query_templates.py` | Natural-language queries for each dataset: 13 UCF-Crime queries, 10 UCF-ARG action phrases (with viewpoint variants), 2 RLVS violence/non-violence queries, 26 TinyVIRAT activity phrases. `build_queries()` dispatches by name. |

---

### 4.7 `scripts/` — CLI Pipeline Entry Points

| File | Pipeline Step | Description |
|---|---|---|
| `extract_embeddings.py` | Step 1 | Batch-encodes dataset clips → `results/embeddings/<model>_<dataset>_embeddings.{npy,json}`. Features: **resume** (skips done clips), **OOM recovery** (auto batch halving), **checkpointing** (saves every 256 clips), **`--limit`** for smoke tests, **`--dataset all`** for all datasets. |
| `build_index.py` | Step 2 | Reads .npy embeddings → builds FAISS index → `results/indices/<model>_<dataset>.faiss` + `.meta.json`. |
| `run_retrieval.py` | Step 3a | Text query → top-K clip retrieval. Loads encoder + index, prints ranked results, optional `--save` to JSON. |
| `generate_report.py` | Step 3b | Retrieves evidence → generates LLM report → computes grounding score → saves to `results/reports/`. Supports `--ungrounded` for H3 control condition. |
| `run_evaluation.py` | Step 4 | Selectable metrics: `--metrics retrieval,efficiency,faithfulness`. Outputs CSVs to `results/metrics/`. |
| `compare_models.py` | Step 5 | VL-JEPA vs CLIP head-to-head across datasets. Outputs `comparison_table.csv` + bar charts in `results/plots/`. |

---

### 4.8 `demo/`

| File | Description |
|---|---|
| `streamlit_app.py` | Interactive web UI (port 8501). Features: model selector (CLIP / VL-JEPA / side-by-side), CLIP variant picker, index selector, Top-K slider, query input, thumbnail grid of results, LLM report generation with grounding score + hallucination rate display. Uses `@st.cache_resource` for caching. |

---

### 4.9 `tests/`

| File | Description |
|---|---|
| `conftest.py` | Shared fixtures: `sample_config` — complete in-memory config using `tmp_path`. No real datasets or model downloads needed. |
| `test_data_loaders.py` | Tests all 4 dataset loaders with synthetic file trees. Validates indexing, counts, labels, and the RLVS AVI gotcha. |
| `test_encoders.py` | Smoke tests for encoder factory and embedding dimension consistency. |
| `test_faithfulness.py` | Tests grounding score, hallucination rate, sentence splitting, and number/date checks. |
| `test_retrieval.py` | Tests FAISS indexer (add/save/load/search round-trip), retriever wiring, and retrieval metrics math. |

---

### 4.10 `utils/`

| File | Description |
|---|---|
| `__init__.py` | Re-exports all helpers from `common.py`. |
| `common.py` | Core utilities: `load_config()` (reads YAML + env overrides), `dataset_root()` / `config_file()` (path resolution), `resolve_device()` (auto→cuda/cpu), `set_seed()` (deterministic seeding), `clear_gpu_cache()`, `ensure_dir()` (mkdir -p), `get_logger()` (timestamped stdout), `PROJECT_ROOT` constant. |

---

### 4.11 Root Deployment Files

| File | Description |
|---|---|
| `Dockerfile` | `python:3.10-slim` image. Installs ffmpeg, CPU PyTorch, all pip deps. Non-root user. Defaults to Streamlit demo on port 8501. |
| `docker-compose.yml` | Two services: **demo** (Streamlit UI, port 8501) and **pipeline** (CLI for scripts). Datasets mounted read-only at `/data`, results bind-mounted, HF cache in named volume. |
| `.env.example` | Template: `CAPSTONE_DATA_ROOT_HOST`, `CAPSTONE_DATA_ROOT`, `CAPSTONE_DEVICE`. |
| `Makefile` | Shortcuts: `setup`, `test`, `extract`, `index`, `retrieve`, `report`, `evaluate`, `compare`, `demo`, `docker-build`, `docker-up`. |
| `requirements.txt` | Dependencies: PyTorch≥2.1, Transformers≥4.44, FAISS-CPU, OpenCV, Streamlit, pandas, scikit-learn, BERTScore, psutil. Conditional: bitsandbytes (Linux), decord (non-Windows). |
| `setup.py` | Package `vljepa-clip-cctv` v1.0.0, Python≥3.10. |
| `.gitignore` | Excludes `__pycache__/`, `.venv/`, `.env`, `results/`, `*.npy`, `*.faiss`, `*.meta.json`. |

---

### 4.12 `reference/` — Research Documents

| File | Description |
|---|---|
| `CCTV_VL-JEPA_Capstone_Research_Report (1).pdf` | Full capstone research report with methodology, experiments, and findings. |
| `Literature_Survey_Report.pdf` | Literature survey on video anomaly detection and vision-language models. |
| `Literature_Survey_report_VL-JEPA_vs_CLIP_for_CCTV_Event_Retrieval.pdf` | Detailed VL-JEPA vs CLIP comparative literature survey. |
| `VL-JEPA_vs_CLIP_for_CCTV_Event_Retrieval.pptx` | Project presentation slides. |
| `DATASET_CARDS.md` | Comprehensive dataset cards for all 4 benchmarks (file counts, sizes, structures, gotchas). |
| `project ppt content text.txt` | Text content from the presentation. |
| `workflow.png` | System workflow diagram. |
| `technicalrequirements.png` | Technical requirements diagram. |

---

## 5. Hypothesis → Code Mapping

| Hypothesis | Implementation | Evaluation |
|---|---|---|
| **H1** (Retrieval) | `models/clip_encoder.py`, `models/vljepa_encoder.py` | `scripts/compare_models.py` → Recall@K, MRR, nDCG, mAP |
| **H2** (Generalization) | `data/ucf_arg_dataset.py` (viewpoint filtering) | `evaluation/generalization_eval.py` → cross-viewpoint recall matrix |
| **H3** (Faithfulness) | `reporting/llm_reporter.py` (grounded vs ungrounded) | `evaluation/faithfulness_metrics.py` → grounding score, hallucination rate |

---

## 6. Key Design Decisions

| Decision | Rationale |
|---|---|
| L2-normalize all embeddings + FAISS `IndexFlatIP` | Inner product on unit vectors = cosine similarity, the standard for VL retrieval |
| Auto-switch to IVF past 20k vectors | Keeps flat-index accuracy for small datasets, scales for large ones |
| Decord with OpenCV fallback | Decord is faster but unavailable on Windows; OpenCV is universal |
| Mean frame pooling for CLIP | CLIP is a per-frame model; temporal mean is a simple strong baseline |
| V-JEPA 2 + CLIP text truncation | Zero-training alignment whose quality is part of what H1 evaluates |
| Flan-T5 for reporting | Lightweight seq2seq LLM; supports quantization on Linux |
| Grounding score = sentence-level overlap | Follows maritime hallucination study methodology from the literature survey |

---

## 7. Pipeline Execution Order

```
Step 1:  python scripts/extract_embeddings.py --model clip --dataset rlvs
Step 2:  python scripts/build_index.py        --model clip --dataset rlvs
Step 3a: python scripts/run_retrieval.py      --model clip --dataset rlvs --query "fight"
Step 3b: python scripts/generate_report.py    --model clip --dataset rlvs --query "fight"
Step 4:  python scripts/run_evaluation.py     --model clip --dataset rlvs --metrics retrieval,efficiency
Step 5:  python scripts/compare_models.py     --models vljepa,clip --datasets ucf_crime,rlvs
Step 6:  streamlit run demo/streamlit_app.py
```

---

## 8. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python ≥ 3.10 |
| Deep Learning | PyTorch ≥ 2.1, Transformers ≥ 4.44 |
| Vision Models | CLIP (ViT-B/32, ViT-L/14), V-JEPA 2 (ViT-S/B/L) |
| Text Model | Flan-T5 (small/base/large) |
| Vector Search | FAISS (FlatIP / IVFFlat) |
| Video Decoding | Decord / OpenCV |
| Web Demo | Streamlit |
| Containerization | Docker + Docker Compose |
| Testing | pytest |
| Metrics | scikit-learn, BERTScore, matplotlib |
