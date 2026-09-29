# Kilocode Agent: Full Codebase Generation Prompt
## Project: VL-JEPA vs. CLIP for CCTV Event Retrieval — Efficiency, Generalization & Faithful Report Generation

---

> [!IMPORTANT]
> This document is a complete, self-contained specification for a Kilocode agent to generate the full project codebase. Read every section carefully before generating any file. All datasets are stored on an **external storage device at `D:\capstone dataset\`** and must never be copied or moved — only read from that path.

---

## 1. Project Overview

**Title**: VL-JEPA vs. CLIP for CCTV Event Retrieval: Efficiency, Generalization, and Faithful Report Generation

**Type**: Engineering Capstone Research Project

**Goal**: Build a lightweight, reproducible framework that:
1. Ingests real CCTV surveillance videos from multiple benchmark datasets
2. Encodes video clips using both **pretrained VL-JEPA** and **pretrained CLIP** vision-language models
3. Indexes all embeddings in a **FAISS vector database**
4. Accepts **natural-language text queries** and retrieves the top-k most relevant video clips (semantic retrieval)
5. Feeds retrieved clips as evidence to a **lightweight LLM** to generate a structured incident report
6. Evaluates the **faithfulness / hallucination rate** of the generated report against actual video evidence
7. Provides a **Streamlit web demo** for interactive querying

**Core Research Questions**:
- H1: Can pretrained VL-JEPA achieve comparable or better semantic retrieval than CLIP for CCTV?
- H2: Does VL-JEPA generalize better than CLIP across different camera viewpoints?
- H3: Does grounding LLM reports in structured visual evidence reduce hallucination?

---

## 2. System Architecture & Pipeline

The system follows an 8-stage pipeline (as documented in the project workflow diagram):

```
Stage 1: CCTV Video Input
    → Raw surveillance video clips loaded from D:\capstone dataset\

Stage 2: Visual Representation
    → Frames / clips extracted and encoded by VL-JEPA encoder OR CLIP visual encoder
    → Both models run in parallel for comparative evaluation

Stage 3: Embedding Generation
    → High-dimensional embedding vectors produced per clip
    → VL-JEPA: joint video-language embedding space
    → CLIP: image-text embedding space (video averaged over sampled frames)

Stage 4: Text Query Input
    → User natural-language query (e.g., "person running away after assault")
    → Encoded by VL-JEPA text encoder OR CLIP text encoder respectively

[Vector Database — Index]
    → FAISS flat or IVF index stores ALL clip embeddings
    → Separate FAISS indices for VL-JEPA and CLIP embeddings

Stage 5: Retrieval
    → Cosine similarity search between text query embedding and all clip embeddings
    → Returns top-k (default k=5) most similar clips

Stage 6: Evidence Collection
    → Top-k retrieved clips surface as evidence with similarity scores and metadata

Stage 7: LLM Report Generation
    → Retrieved clips + metadata fed as structured prompt to lightweight LLM
    → LLM generates a natural-language incident report / summary

Stage 8: Hallucination / Faithfulness Evaluation
    → Generated report evaluated against grounding evidence
    → Metrics: faithfulness score, hallucination rate, consistency with evidence
    → Feedback loop: dashed arrow from evaluation back to retrieval for iterative improvement
```

---

## 3. Datasets (External Storage — `D:\capstone dataset\`)

> [!CAUTION]
> All datasets reside on **`D:\capstone dataset\`** (external storage device). Code must use this path. Do NOT hardcode any other path. Provide a `config.yaml` where this root path is configurable.

### 3.1 UCF-Crime — Anomaly Detection Dataset
- **Path**: `D:\capstone dataset\Anomaly-Detection-Dataset\`
- **Format**: MP4 (H.264), `.txt` annotation files
- **Size**: ~98.82 GB total (partially extracted; Parts 1–4 zip archives + extracted folders)
- **Classes**: 13 anomaly classes (Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism) + Normal
- **Extracted folders available**:
  - `Anomaly-Videos-Part-1/` — 200 videos (Abuse[50], Arrest[50], Arson[50], Assault[50])
  - `Anomaly-Videos-Part-2/` — 74 videos (Burglary[74]; Explosion & Fighting still in zip)
  - `Anomaly-Videos-Part-3/` — 93 videos (Robbery[93]; RoadAccidents & Shooting still in zip)
  - `Anomaly-Videos-Part-4/` — 70 videos (Stealing[20], Vandalism[50]; Shoplifting still in zip)
  - `Normal_Videos_for_Event_Recognition/` — 50 normal clips
- **Annotations**:
  - `Anomaly_Train.txt` — 1,610 training video paths + class labels
  - `Temporal_Anomaly_Annotation_for_Testing_Videos.txt` — 290 test videos with start/end frame pairs (30 FPS)
- **Use in project**: Primary anomaly retrieval benchmark; text queries like "arrest scene" or "arson fire" tested against this dataset
- **Data loader note**: Handle partially-extracted state gracefully — only load what's extracted

### 3.2 UCF-ARG — Multi-View / Cross-Camera Dataset
- **Path**: `D:\capstone dataset\cross camera data\`
- **Format**: AVI
- **Size**: 7.11 GB, 1,410 video clips
- **Directory structure**:
  - `aerial_clips/aerial_clips/<class>/<filename>.avi`
  - `ground_clips/ground_clips/<class>/<filename>.avi`
  - `rooftop_clips/rooftop_clips/<class>/<filename>.avi`
- **Classes (10 actions)**: boxing, carrying, clapping, digging, jogging, openclosetrunk, running, throwing, walking, waving
- **Filename pattern**: `person<ActorID>_<RunID>_<viewpoint>_<action>.avi`
- **Use in project**: Cross-camera generalization evaluation — same action, different viewpoints (aerial vs. ground vs. rooftop). Tests H2.
- **Clip duration**: ~6–10 seconds, ~200–300 frames @ ~30 FPS

### 3.3 RLVS — Real Life Violence Situations Dataset
- **Path**: `D:\capstone dataset\real life voilence data\Real Life Violence Dataset\`
- **Format**: MP4 + AVI (mixed)
- **Size**: 1.81 GB (use only ONE copy — the deduplicated path above)
- **Classes**: Violence (1,000 videos: `V_1.mp4` … `V_1000.mp4`), NonViolence (1,000 videos: `NV_1` … `NV_1000`)
- **Critical**: NonViolence class has **49 `.avi` files** (NV_602.avi, NV_863.avi–NV_911.avi). Data loader MUST glob both `*.mp4` AND `*.avi` or it will silently drop 49 samples.
- **Use in project**: Violence detection retrieval; binary classification queries ("violent fight", "normal street activity")

### 3.4 TinyVIRAT-v2 — Low-Resolution Surveillance Action Dataset
- **Path**: `D:\capstone dataset\tinyvirat\TinyVIRAT-v2\TinyVIRAT_V2\`
- **Format**: MP4 (H.264), JSON metadata
- **Size**: 2.04 GB total (both v1 + v2), use v2 for experiments
- **Splits**:
  - `videos/train/` — 16,950 clips
  - `videos/val/` — 3,308 clips
  - `videos/test/` — 6,097 clips (named `00000.mp4` … `06096.mp4`)
- **Metadata files**:
  - `class_map.json` — 26-class index mapping
  - `tiny_train_v2.json`, `tiny_val_v2.json`, `tiny_test_v2_public.json`
- **Annotation schema**:
  ```json
  {
    "id": "000000",
    "video_id": "VIRAT_S_000204_04_000738_000977",
    "path": "VIRAT_S_000204_04_000738_000977/000000.mp4",
    "dim": [82, 72, 72],
    "label": ["activity_carrying", "activity_walking"]
  }
  ```
- **26 classes**: Opening, Interacts, Pull, activity_carrying, Entering, vehicle_moving, Exiting, Loading, Talking, activity_running, vehicle_turning_left, vehicle_stopping, Riding, Closing, activity_walking, Push, specialized_using_tool, vehicle_starting, specialized_miscellaneous, activity_standing, Transport_HeavyCarry, activity_gesturing, vehicle_turning_right, specialized_talking_phone, specialized_texting_phone, Misc
- **Special note**: Native spatial resolution ~72×72 pixels. Use LOW pooling strides. Do NOT apply heavy spatial downsampling. Consider spatio-temporal attention (TimeSformer/Video Swin style).
- **Use in project**: Multi-label low-resolution retrieval; tests model robustness on tiny subjects

---

## 4. Models & Technical Specifications

### 4.1 VL-JEPA (Primary Novel Model)
- **Full name**: Vision-Language Joint Embedding Predictive Architecture
- **Paper**: arXiv:2512.10942 — Meta FAIR, HKUST, Sorbonne Université, NYU (2025)
- **Total Parameters**: ~1.6 billion

#### Four-Component Architecture
| Component | Description |
|---|---|
| **X-Encoder** | Frozen **V-JEPA 2 ViT-L** (Vision Transformer Large) — compresses video/image frames into stable latent representations. Frozen during training (not fine-tuned). |
| **Y-Encoder** | Converts target text into embeddings during training; initialized from **EmbeddingGemma** |
| **Predictor** | Initialized from **Llama 3** transformer layers; takes visual embeddings + textual query → predicts target embedding in abstract representation space |
| **Text Decoder** | Lightweight optional module — invoked only at inference when human-readable text output is required |

#### Key Properties
- **Non-autoregressive**: predicts continuous embeddings instead of generating tokens sequentially
- **~50% fewer trainable parameters** than comparable token-space VLMs (X-Encoder is frozen)
- **~2.85× reduction** in decoding operations via selective/lazy decoding
- Unified embedding space: handles text-to-video retrieval, open-vocabulary classification, discriminative VQA without structural modification per task
- Abstracts away pixel-level noise (illumination, weather, blur) — especially relevant for CCTV footage quality degradation
- Outperforms CLIP, SigLIP2, and Perception Encoder on video retrieval benchmarks (MSR-VTT, ActivityNet)

#### Implementation
- **Pretrained checkpoint**: Official Meta GitHub + HuggingFace — `facebook/vl-jepa` or `facebook/vjepa2-vitl-fpc64-256`
  - If VL-JEPA not separately available, use **V-JEPA 2** (ViT-L) as the visual encoder + pair with a text encoder
  - Fallback: Use `facebook/jepa` GitHub repo with pretrained weights
- **Input**: Video clip → sample N=8 or N=16 frames uniformly → resize to 224×224 → apply ViT-L normalization (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`)
- **Output**: Single L2-normalized embedding vector per clip (dim=1024 for ViT-L)
- **Text input**: EmbeddingGemma tokenizer → Y-Encoder → normalized text embedding
- **Special handling for TinyVIRAT**: Do NOT downsample below native ~72×72 resolution before encoding; apply center-padding to 224×224 or use super-resolution preprocessing

### 4.2 CLIP (Baseline Comparison Model)
- **Full name**: Contrastive Language-Image Pre-Training
- **Variants to use**:
  - Primary: `openai/clip-vit-large-patch14` (ViT-L/14) — strongest CLIP baseline
  - Secondary: `openai/clip-vit-base-patch32` (ViT-B/32) — lighter, for resource-constrained comparison
- **Video handling strategy** (CLIP is image-based):
  - Sample N frames uniformly (N=8 default) from each video clip
  - Encode each frame independently through CLIP visual encoder
  - Average the frame embeddings → single clip-level embedding
  - This is the standard "mean frame pooling" approach for CLIP on video
- **Input**: Each frame resized to 224×224, normalized with CLIP-specific mean/std
- **Output**: Normalized embedding vector (dimension = 768 for ViT-L/14)
- **Text input**: CLIP tokenizer → CLIP text encoder → normalized text embedding
- **Library**: `transformers` (HuggingFace) or `openai/clip` PyPI package

### 4.3 VL-JEPA vs. CLIP — Architecture Comparison Table

| Property | CLIP | VL-JEPA |
|---|---|---|
| **Learning Method** | Contrastive learning (image/video ↔ text alignment) | Joint Embedding Predictive Architecture (latent-space prediction) |
| **Primary Design** | Image-text pairs | Video + language in shared embedding space |
| **Autoregressive** | No | No (prediction-based, not generative) |
| **Visual Backbone** | ViT-B/32 or ViT-L/14 | Frozen V-JEPA 2 ViT-L |
| **Text Backbone** | Transformer text encoder | EmbeddingGemma |
| **Parameters** | ~400M (ViT-L/14) | ~1.6B total (~50% fewer trainable) |
| **Decoding Speed** | N/A | 2.85× faster than autoregressive VLMs |
| **CCTV Suitability** | Flexible but image-centric; may miss temporal patterns | Abstracts away pixel noise (lighting, blur, weather) via latent prediction |
| **Training Requirement** | Massive image-text pairs | Self-supervised; focuses on task-relevant semantics |
| **Video Handling** | Frame-by-frame + mean pooling | Native video tube encoding |
| **Benchmark** | Strong on image retrieval | Outperforms CLIP on MSR-VTT, ActivityNet video retrieval |

### 4.4 LLM for Report Generation
- **Purpose**: Generate natural-language incident reports from retrieved CCTV evidence
- **Recommended models** (lightweight, can run on CPU or T4 GPU):
  - Primary: `google/flan-t5-large` or `google/flan-t5-xl` (instruction-tuned T5)
  - Alternative: `mistralai/Mistral-7B-Instruct-v0.2` (4-bit quantized via bitsandbytes)
  - Alternative: `meta-llama/Llama-3.2-3B-Instruct` (smallest Llama 3 instruct)
  - Fallback: OpenAI `gpt-3.5-turbo` via API (if local GPU insufficient)
- **Report generation prompt template** (feed retrieved evidence as structured context):
  ```
  You are a surveillance analyst. Based on the following retrieved CCTV evidence clips, 
  write a concise factual incident report. Only include information directly supported 
  by the evidence. Do not add speculation.

  Evidence:
  - Clip 1: [filename], similarity score: [score], dataset: [dataset], label: [label]
    Description: [auto-generated frame description]
  - Clip 2: ...
  ...

  Generate a structured incident report with: Incident Type, Time/Location (if available), 
  Observed Behaviors, Severity Assessment, and Recommended Action.
  ```
- **Hallucination mitigation**: Use retrieval-augmented generation (RAG) — LLM must stay grounded to evidence context

---

## 5. Technical Stack & Dependencies

### 5.1 Environment
```
Python: 3.10+
CUDA: 11.8 or 12.1 (if NVIDIA GPU available)
OS: Windows 10/11 or Ubuntu Linux
```

### 5.2 Core Dependencies (`requirements.txt`)
```
# Deep Learning
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0

# Vision-Language Models
transformers>=4.40.0
timm>=0.9.0
open-clip-torch>=2.24.0

# VL-JEPA (use official Meta implementation if not on HF)
# git+https://github.com/facebookresearch/jepa.git

# Video Processing
opencv-python>=4.9.0
ffmpeg-python>=0.2.0
decord>=0.6.0        # fast video loading

# Vector Search
faiss-cpu>=1.7.4     # or faiss-gpu if CUDA available
numpy>=1.26.0

# LLM
accelerate>=0.27.0
bitsandbytes>=0.43.0  # for 4-bit quantization
sentencepiece>=0.1.99

# Evaluation
scikit-learn>=1.4.0
scipy>=1.12.0

# Faithfulness/Hallucination Evaluation
bert-score>=0.3.13
evaluate>=0.4.1

# Experiment Tracking
wandb>=0.16.0        # optional

# Demo
streamlit>=1.34.0
plotly>=5.20.0

# Utilities
tqdm>=4.66.0
pyyaml>=6.0.1
pandas>=2.2.0
pillow>=10.3.0
matplotlib>=3.8.0
seaborn>=0.13.0
jupyter>=1.0.0
```

### 5.3 Hardware Requirements
| Component | Minimum | Recommended |
|---|---|---|
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| GPU | NVIDIA 8 GB VRAM | NVIDIA 16 GB+ (A100/T4) |
| RAM | 8 GB | 16 GB |
| Storage | 50 GB free | 100 GB free |
| Dataset Storage | External `D:\` drive | External SSD |

---

## 6. Project Directory Structure (to be generated)

```
capstone/
├── config/
│   └── config.yaml                    # All paths, hyperparameters, model names
│
├── data/
│   ├── __init__.py
│   ├── base_dataset.py                # Abstract base class for all datasets
│   ├── ucf_crime_dataset.py           # UCF-Crime loader
│   ├── ucf_arg_dataset.py             # UCF-ARG multi-view loader
│   ├── rlvs_dataset.py                # RLVS violence dataset loader
│   ├── tinyvirat_dataset.py           # TinyVIRAT-v2 loader
│   └── data_utils.py                  # Frame sampling, video decoding utilities
│
├── models/
│   ├── __init__.py
│   ├── vljepa_encoder.py              # VL-JEPA video + text encoder wrapper
│   ├── clip_encoder.py                # CLIP video + text encoder wrapper
│   └── encoder_factory.py             # Factory to instantiate either model
│
├── retrieval/
│   ├── __init__.py
│   ├── embedding_indexer.py           # FAISS index build, save, load
│   ├── retriever.py                   # Text query → top-k retrieval
│   └── retrieval_utils.py             # Cosine similarity, normalization helpers
│
├── reporting/
│   ├── __init__.py
│   ├── llm_reporter.py                # LLM-based incident report generator
│   └── prompt_templates.py            # Report generation prompt templates
│
├── evaluation/
│   ├── __init__.py
│   ├── retrieval_metrics.py           # Recall@K, MRR, nDCG, mAP
│   ├── faithfulness_metrics.py        # BERTScore, faithfulness, hallucination rate
│   ├── efficiency_metrics.py          # Latency, memory, throughput
│   └── generalization_eval.py         # Cross-camera / cross-dataset eval (H2)
│
├── scripts/
│   ├── extract_embeddings.py          # Batch embed all dataset clips → save to disk
│   ├── build_index.py                 # Build FAISS index from saved embeddings
│   ├── run_retrieval.py               # CLI: query → top-k results
│   ├── generate_report.py             # CLI: evidence → LLM report
│   ├── run_evaluation.py              # Full evaluation pipeline
│   └── compare_models.py              # Head-to-head VL-JEPA vs CLIP comparison
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Dataset EDA, frame samples, statistics
│   ├── 02_embedding_visualization.ipynb  # UMAP/t-SNE of embeddings
│   ├── 03_retrieval_demo.ipynb        # Interactive retrieval examples
│   ├── 04_report_generation.ipynb     # LLM report generation examples
│   └── 05_results_analysis.ipynb      # Final results, plots, tables
│
├── demo/
│   └── streamlit_app.py               # Streamlit web demo
│
├── tests/
│   ├── test_data_loaders.py
│   ├── test_encoders.py
│   ├── test_retrieval.py
│   └── test_faithfulness.py
│
├── results/
│   ├── embeddings/                    # Saved .npy / .pt embedding files
│   ├── indices/                       # Saved FAISS index files
│   ├── reports/                       # Generated LLM incident reports
│   └── metrics/                       # Evaluation result CSVs and plots
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 7. Configuration File (`config/config.yaml`)

```yaml
# ============================================================
# Master Configuration — VL-JEPA vs CLIP CCTV Retrieval
# ============================================================

# ----- Data Paths (External Storage) -----
data:
  root: "D:/capstone dataset"
  ucf_crime:
    path: "D:/capstone dataset/Anomaly-Detection-Dataset"
    train_split: "D:/capstone dataset/Anomaly-Detection-Dataset/Anomaly_Train.txt"
    test_annotations: "D:/capstone dataset/Anomaly-Detection-Dataset/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
    extracted_parts:
      - "Anomaly-Videos-Part-1"
      - "Anomaly-Videos-Part-2"
      - "Anomaly-Videos-Part-3"
      - "Anomaly-Videos-Part-4"
      - "Normal_Videos_for_Event_Recognition"
  ucf_arg:
    path: "D:/capstone dataset/cross camera data"
    viewpoints: ["aerial_clips", "ground_clips", "rooftop_clips"]
    classes: ["boxing", "carrying", "clapping", "digging", "jogging",
              "openclosetrunk", "running", "throwing", "walking", "waving"]
  rlvs:
    path: "D:/capstone dataset/real life voilence data/Real Life Violence Dataset"
    classes: ["Violence", "NonViolence"]
    extensions: ["*.mp4", "*.avi"]   # IMPORTANT: must include *.avi
  tinyvirat:
    path: "D:/capstone dataset/tinyvirat/TinyVIRAT-v2/TinyVIRAT_V2"
    version: "v2"
    class_map: "class_map.json"
    train_json: "tiny_train_v2.json"
    val_json: "tiny_val_v2.json"
    test_json: "tiny_test_v2_public.json"

# ----- Video Processing -----
video:
  num_frames: 8          # frames sampled per clip
  frame_size: 224        # spatial resolution (H=W)
  fps: 30                # assumed FPS for temporal annotation
  tinyvirat_min_size: 72 # do NOT resize below native resolution for TinyVIRAT

# ----- Model Configuration -----
models:
  vljepa:
    name: "facebook/vl-jepa"
    checkpoint: null     # set to local path if downloaded manually
    embedding_dim: 768
    device: "cuda"       # or "cpu"
    batch_size: 8
  clip:
    primary: "openai/clip-vit-large-patch14"
    secondary: "openai/clip-vit-base-patch32"
    embedding_dim: 768
    device: "cuda"
    batch_size: 16
    video_strategy: "mean_frame_pooling"  # average N frame embeddings

# ----- Retrieval -----
retrieval:
  index_type: "FlatIP"   # FAISS IndexFlatIP (inner product = cosine on normalized vecs)
  top_k: 5
  embedding_save_dir: "results/embeddings"
  index_save_dir: "results/indices"

# ----- LLM Report Generation -----
llm:
  model: "google/flan-t5-large"   # or mistralai/Mistral-7B-Instruct-v0.2
  quantization: "4bit"            # none | 4bit | 8bit
  max_new_tokens: 512
  temperature: 0.3
  device: "cuda"

# ----- Evaluation -----
evaluation:
  recall_k_values: [1, 5, 10]
  mrr: true
  ndcg: true
  latency_runs: 50          # number of queries for latency measurement
  faithfulness_metric: "bertscore"
  hallucination_threshold: 0.5

# ----- Experiment -----
experiment:
  name: "vljepa_vs_clip_cctv"
  seed: 42
  log_wandb: false
  output_dir: "results"
```

---

## 8. Detailed Module Specifications

### 8.1 `data/base_dataset.py`
Abstract PyTorch Dataset class:
- `__init__(self, config)`: read config, set paths
- `__len__(self)`: return number of clips
- `__getitem__(self, idx)`: return `{"video": Tensor[T,C,H,W], "label": str/list, "path": str, "metadata": dict}`
- Abstract methods: `_load_video(path)`, `_get_labels(idx)`, `_build_index()`

### 8.2 `data/ucf_crime_dataset.py`
- Scan only EXTRACTED folders (check existence before adding to index)
- Parse `Anomaly_Train.txt` for class labels
- Parse `Temporal_Anomaly_Annotation_for_Testing_Videos.txt` for test temporal annotations
- Support modes: `train`, `test`, `all`
- Return class label as string from the 13 anomaly classes + "Normal"

### 8.3 `data/ucf_arg_dataset.py`
- Scan all three viewpoint subdirectories
- Parse filename to extract: `actor_id`, `run_id`, `viewpoint`, `action`
- Support filtering by viewpoint for cross-camera evaluation
- Return metadata including `viewpoint` field for H2 generalization analysis

### 8.4 `data/rlvs_dataset.py`
- Use `glob.glob("*.mp4") + glob.glob("*.avi")` — never just `*.mp4`
- Only use path: `D:\capstone dataset\real life voilence data\Real Life Violence Dataset\`
- Classes: `Violence/` and `NonViolence/` subdirectories
- Return binary label: `violence` or `non_violence`

### 8.5 `data/tinyvirat_dataset.py`
- Load from JSON metadata files (not folder scan)
- Support multi-label output (list of class strings)
- **Special**: For clips with native dim ≤ 72×72, use `frame_size = min(72, config.video.frame_size)` — do NOT upscale aggressively
- Return multi-label tensor for evaluation

### 8.6 `data/data_utils.py`
- `sample_frames(video_path, num_frames, backend='decord')`: uniformly sample N frames from a video
- `decode_video(video_path, backend='decord')`: full video decode with fallback to OpenCV
- `preprocess_frame(frame, size, mean, std)`: resize + normalize
- `collate_fn(batch)`: handles variable-length clips

### 8.7 `models/vljepa_encoder.py`
```python
class VLJEPAEncoder:
    def __init__(self, config): ...
    def encode_video(self, video_tensor: Tensor) -> Tensor:
        # video_tensor: [B, T, C, H, W]
        # returns: [B, D] normalized embeddings
    def encode_text(self, text: List[str]) -> Tensor:
        # returns: [B, D] normalized embeddings
    def get_embedding_dim(self) -> int: ...
```

### 8.8 `models/clip_encoder.py`
```python
class CLIPEncoder:
    def __init__(self, config, variant='primary'): ...
    def encode_video(self, video_tensor: Tensor) -> Tensor:
        # video_tensor: [B, T, C, H, W]
        # encode each frame → average → L2 normalize
        # returns: [B, D] normalized embeddings
    def encode_text(self, text: List[str]) -> Tensor:
        # returns: [B, D] normalized embeddings
    def get_embedding_dim(self) -> int: ...
```

### 8.9 `retrieval/embedding_indexer.py`
```python
class EmbeddingIndexer:
    def __init__(self, embedding_dim, index_type='FlatIP'): ...
    def add(self, embeddings: np.ndarray, metadata: List[dict]): ...
    def save(self, path: str): ...
    def load(self, path: str): ...
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, List[dict]]:
        # returns distances and metadata for top-k results
```
- Store metadata (clip path, label, dataset name, viewpoint) alongside FAISS index in a parallel JSON file

### 8.10 `retrieval/retriever.py`
```python
class CCTVRetriever:
    def __init__(self, encoder, indexer): ...
    def retrieve(self, text_query: str, k: int = 5) -> List[dict]:
        # 1. Encode text query
        # 2. Search FAISS index
        # 3. Return top-k results with metadata + scores
```

### 8.11 `reporting/llm_reporter.py`
```python
class IncidentReporter:
    def __init__(self, config): ...
    def generate_report(self, evidence: List[dict], query: str) -> str:
        # Format evidence into structured prompt
        # Call LLM
        # Return generated report string
    def _format_evidence_prompt(self, evidence: List[dict], query: str) -> str: ...
```

### 8.12 `evaluation/retrieval_metrics.py`
Implement these functions:
- `recall_at_k(retrieved_labels, relevant_labels, k)` → float
- `mean_reciprocal_rank(results_list)` → float
- `ndcg_at_k(retrieved_labels, relevant_labels, k)` → float
- `mean_average_precision(results_list)` → float
- `compute_all_metrics(results, ground_truth, k_values=[1, 5, 10])` → dict

### 8.13 `evaluation/faithfulness_metrics.py`
Implement:
- `bertscore_faithfulness(generated_report, evidence_text)` → float (0–1)
- `hallucination_rate(generated_report, evidence_text)` → float (fraction of unsupported sentences)
- `compute_faithfulness_report(reports, evidence_list)` → DataFrame

### 8.14 `evaluation/efficiency_metrics.py`
- `measure_encoding_latency(encoder, sample_clips, n_runs=50)` → dict with mean/std latency
- `measure_memory_usage(encoder, sample_clip)` → dict with peak GPU/CPU memory
- `measure_throughput(encoder, dataset_loader)` → clips_per_second

### 8.15 `evaluation/generalization_eval.py`
For H2 (cross-camera generalization):
- Train on one viewpoint (e.g., `ground_clips`), test on others (`aerial_clips`, `rooftop_clips`)
- Compute Recall@K for each viewpoint combination
- Plot cross-viewpoint heatmap

### 8.16 `scripts/extract_embeddings.py`
CLI script:
```
python scripts/extract_embeddings.py \
    --model vljepa \        # or clip
    --dataset ucf_crime \   # or ucf_arg, rlvs, tinyvirat, all
    --config config/config.yaml \
    --output results/embeddings/
```
- Saves embeddings as `.npy` files + metadata JSON per dataset per model
- Shows tqdm progress bar
- Handles GPU OOM gracefully with batch size reduction

### 8.17 `scripts/run_evaluation.py`
CLI script:
```
python scripts/run_evaluation.py \
    --model vljepa \
    --dataset ucf_crime \
    --config config/config.yaml \
    --metrics all
```
- Outputs results to `results/metrics/<model>_<dataset>_metrics.csv`
- Prints summary table to console

### 8.18 `scripts/compare_models.py`
- Runs both VL-JEPA and CLIP on all datasets
- Generates comparison table (LaTeX + CSV)
- Generates bar charts: Recall@1, Recall@5, Recall@10, MRR, Latency, Memory
- Saves all plots to `results/plots/`

---

## 9. Streamlit Demo (`demo/streamlit_app.py`)

Features:
1. **Sidebar**: Model selector (VL-JEPA / CLIP / Both), Dataset selector, Top-K slider
2. **Main area**: Text query input box
3. **Results panel**: 
   - Top-k retrieved clips shown as video thumbnails (first frame) with similarity score
   - Clip metadata (dataset, label, viewpoint)
4. **Report panel**: LLM-generated incident report displayed with faithfulness score
5. **Comparison mode**: Side-by-side VL-JEPA vs CLIP results when "Both" selected
6. **Metrics panel**: Real-time latency display

Run command: `streamlit run demo/streamlit_app.py`

---

## 10. Evaluation Protocol

### 10.1 Retrieval Evaluation
For each model (VL-JEPA, CLIP-L/14, CLIP-B/32) × each dataset:
- Create text queries from class labels (e.g., label `"Assault"` → query `"person being assaulted"`)
- Use class-based relevance: retrieved clip is relevant if its class matches query class
- Compute: Recall@1, Recall@5, Recall@10, MRR, nDCG@10

**Evaluation text queries to generate per dataset**:
- UCF-Crime: one query per anomaly class (13 queries)
- UCF-ARG: one query per action class × per viewpoint (30 queries)
- RLVS: 2 queries ("violent physical fight", "normal non-violent activity")
- TinyVIRAT-v2: one query per of the 26 classes

### 10.2 Cross-Camera Generalization (H2)
- Index UCF-ARG ground clips, query with aerial/rooftop clip-derived text
- Index one viewpoint, test on the others
- Compare Recall@5 drop between VL-JEPA and CLIP

### 10.3 Efficiency Evaluation
- Measure per-clip encoding time (mean ± std over 50 runs)
- Measure peak GPU memory during batch encoding
- Measure FAISS search latency
- Report: clips/second throughput

### 10.4 Faithfulness Evaluation (H3)
Two conditions:
1. **Grounded**: LLM receives retrieved evidence context (RAG-style)
2. **Ungrounded**: LLM receives only the text query (no evidence)

Metrics:
- BERTScore F1 between generated report and evidence captions
- Sentence-level entailment (NLI): what fraction of report sentences are supported by evidence?
- Hallucination rate = unsupported_sentences / total_sentences

---

## 11. Notebooks Specification

### `01_data_exploration.ipynb`
- Print statistics for each dataset (total clips, class distribution, avg duration)
- Show sample frames from each dataset
- Plot class distribution bar charts

### `02_embedding_visualization.ipynb`
- Load saved embeddings for UCF-Crime and RLVS
- Run UMAP dimensionality reduction (2D)
- Color-code by class label
- Compare VL-JEPA vs CLIP embedding cluster quality visually

### `03_retrieval_demo.ipynb`
- Interactive: input text query → show top-5 retrieved clips
- Display first frame of each retrieved clip
- Show similarity score

### `04_report_generation.ipynb`
- Pick a query + retrieved evidence set
- Run LLM report generation (grounded vs ungrounded)
- Compare outputs side-by-side

### `05_results_analysis.ipynb`
- Load all metrics CSVs
- Generate final comparison tables (formatted for research paper)
- Generate bar charts, line plots, heatmaps
- Compute statistical significance (paired t-test)

---

## 12. README.md Specification

Must include:
1. Project title and one-line description
2. System requirements
3. Installation instructions (conda env or pip)
4. Dataset setup (note: data on D:\ drive, user must set path in config.yaml)
5. Quick start: extract embeddings → build index → run demo
6. Full pipeline commands
7. Evaluation commands
8. Project structure tree
9. Research hypotheses and expected outcomes
10. Citation section

---

## 13. Key Implementation Rules & Constraints

1. **ALL dataset paths** must come from `config/config.yaml` — no hardcoded paths anywhere
2. **RLVS loader** MUST use `glob("*.mp4") + glob("*.avi")` — never only `*.mp4`
3. **TinyVIRAT** clips must NOT be aggressively downsampled below ~72×72 native resolution
4. **UCF-Crime** loader must gracefully skip categories that are still in ZIP archives (not extracted)
5. **Embeddings are expensive to compute** — always save to disk after batch extraction; provide resume capability (skip already-processed clips)
6. **Both models must use L2-normalized embeddings** for fair cosine similarity comparison
7. **FAISS index type**: `IndexFlatIP` (inner product on L2-normalized vectors = cosine similarity)
8. **Batch processing**: Use DataLoader with `num_workers=4` for efficient video loading
9. **Error handling**: Corrupted video files are common in surveillance datasets — wrap video decode in try/except and log skipped files
10. **Reproducibility**: Set all random seeds from config (`seed: 42`); log all hyperparameters
11. **Memory management**: Clear GPU cache between model switches; use `torch.no_grad()` for all inference
12. **Report grounding**: LLM must only receive evidence from the retrieval system — never allow unconstrained generation without evidence context in the main pipeline

---

## 14. Expected Outputs & Results

The system should produce:
1. `results/embeddings/vljepa_ucf_crime_embeddings.npy` + `.json` (and similar for all model×dataset combinations)
2. `results/indices/vljepa_ucf_crime.faiss` (and similar)
3. `results/metrics/comparison_table.csv` — full Recall@K, MRR, Latency, Memory for both models
4. `results/metrics/faithfulness_report.csv` — grounded vs. ungrounded hallucination rates
5. `results/plots/recall_comparison.png`, `latency_comparison.png`, `embedding_umap.png`
6. `results/reports/` — sample LLM-generated incident reports (JSON format)
7. Running Streamlit demo

---

## 15. Research Contribution Summary

This capstone project contributes:
1. **First systematic comparison** of VL-JEPA vs CLIP for CCTV-specific event retrieval
2. **Cross-camera generalization benchmark** using UCF-ARG multi-view dataset
3. **Faithfulness evaluation framework** for LLM-generated surveillance reports
4. **Lightweight reproducible pipeline** that runs on a single consumer GPU (8 GB VRAM)
5. **Open-source codebase** with all datasets accessible via external drive path configuration

---

*Document generated for Kilocode Agent — Engineering Capstone Project 2026*
*Data storage: External drive at `D:\capstone dataset\`*
*All four datasets fully documented with exact paths, formats, and loading constraints.*
