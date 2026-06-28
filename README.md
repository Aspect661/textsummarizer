# Text Summarizer — End-to-End MLOps Pipeline

An end-to-end MLOps project that fine-tunes Google's **PEGASUS** model on the **SAMSum** dialogue summarization dataset and serves predictions via a **FastAPI** REST API. The project is structured as a 4-stage modular ML pipeline with full configuration management, structured logging, and Docker support.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Pipeline Stages](#pipeline-stages)
- [Configuration](#configuration)
- [Training Parameters](#training-parameters)
- [API Endpoints](#api-endpoints)
- [Setup & Installation](#setup--installation)
- [Running the Pipeline](#running-the-pipeline)
- [Running the API](#running-the-api)
- [Docker](#docker)
- [Key Design Decisions](#key-design-decisions)

---

## Overview

| Item | Detail |
|------|--------|
| **Model** | `google/pegasus-cnn_dailymail` (abstractive summarization) |
| **Dataset** | SAMSum — chat/dialogue pairs with human-written summaries |
| **Task** | Abstractive text summarization of dialogues |
| **Evaluation** | ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum |
| **Serving** | FastAPI on port `8080` |
| **Containerization** | Docker |

---

## Architecture

```
config/config.yaml  ──►  ConfigurationManager  ──►  Typed Config Dataclasses
params.yaml         ──►  ConfigurationManager  ──/

Typed Config Dataclasses
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Data Ingestion                                │
│    Download ZIP → Extract → artifacts/data_ingestion/   │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│  Stage 2: Data Transformation                           │
│    Load SAMSum → Tokenize (PEGASUS tokenizer)           │
│    → Save as HF Dataset → artifacts/data_transformation/│
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│  Stage 3: Model Training                                │
│    Load PEGASUS → Fine-tune on SAMSum (test split)      │
│    → Save model + tokenizer → artifacts/model_trainer/  │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│  Stage 4: Model Evaluation                              │
│    Load fine-tuned model → Run on test[0:10]            │
│    → Compute ROUGE → Save metrics.csv                   │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
                    FastAPI (app.py)
               POST /predict  ──►  PredictionPipeline
               GET  /train    ──►  Triggers main.py
```

---

## Project Structure

```
textsummarizer/
├── config/
│   └── config.yaml                          # All artifact paths & model/data URLs
├── src/
│   └── textSummarizer/
│       ├── constants/
│       │   └── __init__.py                  # CONFIG_FILE_PATH, PARAMS_FILE_PATH
│       ├── entity/
│       │   └── __init__.py                  # Typed dataclasses for all stage configs
│       ├── utils/
│       │   └── common.py                    # read_yaml(), create_directories()
│       ├── logging/
│       │   └── __init__.py                  # Structured logger → logs/continuous_logs.log
│       ├── config/
│       │   └── configuration.py             # ConfigurationManager — reads YAML → dataclasses
│       ├── components/
│       │   ├── data_ingestion.py            # Download + unzip dataset
│       │   ├── data_transformation.py       # Tokenization with PEGASUS tokenizer
│       │   ├── model_trainer.py             # Fine-tune PEGASUS with HF Trainer
│       │   └── model_evaluation.py          # ROUGE scoring on test set
│       └── pipeline/
│           ├── stage_1_data_ingestion_pipeline.py
│           ├── stage_2_data_transformation_pipeline.py
│           ├── stage_3_model_trainer_pipeline.py
│           ├── stage_4_model_evaluation.py
│           └── predicition_pipeline.py      # Inference pipeline for the API
├── research/
│   ├── 1_data_ingestion.ipynb
│   ├── 2_data_transformation.ipynb
│   ├── 3_model_trainer.ipynb
│   └── 4_model_evaluation.ipynb             # Notebooks for iterative development
├── app.py                                   # FastAPI application
├── main.py                                  # Runs all 4 pipeline stages sequentially
├── params.yaml                              # HuggingFace TrainingArguments
├── requirements.txt
├── Dockerfile
└── template.py                              # Project scaffolding script
```

---

## Pipeline Stages

### Stage 1 — Data Ingestion

**Component:** `src/textSummarizer/components/data_ingestion.py`

- Downloads a ZIP archive from GitHub (`summarizer-data.zip`) containing the SAMSum dataset in HuggingFace `datasets` format.
- Skips download if `artifacts/data_ingestion/data.zip` already exists (idempotent).
- Extracts the archive into `artifacts/data_ingestion/`, producing the `samsum_dataset/` directory.

**Artifacts produced:**
```
artifacts/data_ingestion/
├── data.zip
└── samsum_dataset/
    ├── train/
    ├── validation/
    └── test/
```

---

### Stage 2 — Data Transformation

**Component:** `src/textSummarizer/components/data_transformation.py`

- Loads the raw SAMSum HuggingFace dataset from disk.
- Tokenizes both the `dialogue` (input) and `summary` (target) columns using the `google/pegasus-cnn_dailymail` tokenizer.
  - Input max length: **1024 tokens** (truncated)
  - Target max length: **128 tokens** (truncated)
- Uses `text_target=` for encoding the summary (required by seq2seq tokenizers).
- Applies tokenization in batched mode across all splits (train / validation / test).
- Saves the tokenized dataset back to disk as a HuggingFace `DatasetDict`.

**Artifacts produced:**
```
artifacts/data_transformation/
└── samsum_dataset/
    ├── train/
    ├── validation/
    └── test/
```

---

### Stage 3 — Model Training

**Component:** `src/textSummarizer/components/model_trainer.py`

- Loads `google/pegasus-cnn_dailymail` weights from HuggingFace Hub.
- Fine-tunes on the **test split** of the tokenized SAMSum dataset (uses test for training, validation for eval — mirrors the original course setup).
- Uses `DataCollatorForSeq2Seq` for dynamic padding within each batch.
- Saves the fine-tuned model and tokenizer to disk after training.

**Training setup:**
- `gradient_accumulation_steps=16` effectively simulates a larger batch size on limited GPU memory.
- `per_device_train_batch_size=1` — designed for single-GPU / CPU training.
- `warmup_steps=500` — linear learning rate warmup before decay.
- Model saves are effectively disabled (`save_steps=1000000`) so only the final model is kept.

**Artifacts produced:**
```
artifacts/model_trainer/
├── pegasus-samsum-model/      # Fine-tuned model weights
└── tokenizer/                 # Saved tokenizer
```

---

### Stage 4 — Model Evaluation

**Component:** `src/textSummarizer/components/model_evaluation.py`

- Loads the fine-tuned model and tokenizer from `artifacts/model_trainer/`.
- Runs inference on a **10-sample slice** of the test set (`test[0:10]`) for speed.
- Generates summaries using beam search:
  - `num_beams=8`
  - `length_penalty=0.8` (penalizes longer sequences)
  - `max_length=128`
- Computes ROUGE scores against the ground-truth `summary` column using the HuggingFace `evaluate` library.
- Saves scores to a CSV file.

**Metrics computed:**

| Metric | Description |
|--------|-------------|
| `rouge1` | Unigram overlap |
| `rouge2` | Bigram overlap |
| `rougeL` | Longest common subsequence |
| `rougeLsum` | LCS at summary level |

**Artifacts produced:**
```
artifacts/model_evaluation/
└── metrics.csv
```

---

## Configuration

All paths and model references live in `config/config.yaml`. No hardcoded paths exist anywhere in the source code — everything flows through `ConfigurationManager`.

```yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  source_URL: https://github.com/krishnaik06/datasets/raw/refs/heads/main/summarizer-data.zip
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion

data_transformation:
  root_dir: artifacts/data_transformation
  data_path: artifacts/data_ingestion/samsum_dataset
  tokenizer_name: google/pegasus-cnn_dailymail

model_trainer:
  root_dir: artifacts/model_trainer
  data_path: artifacts/data_transformation/samsum_dataset
  model_ckpt: google/pegasus-cnn_dailymail

model_evaluation:
  root_dir: artifacts/model_evaluation
  data_path: artifacts/data_transformation/samsum_dataset
  model_path: artifacts/model_trainer/pegasus-samsum-model
  tokenizer_path: artifacts/model_trainer/tokenizer
  metric_file_name: artifacts/model_evaluation/metrics.csv
```

### How Configuration Flows

```
config/config.yaml
       │
       ▼
ConfigurationManager.read_yaml()  →  ConfigBox (dot-access dict)
       │
       ▼
get_*_config()  →  Typed @dataclass (entity/__init__.py)
       │
       ▼
Component.__init__(config: SomeConfig)
```

`ConfigBox` (from `python-box`) enables dot-notation access on the parsed YAML (e.g., `config.data_ingestion.root_dir`). `@ensure_annotations` enforces type correctness at the utility function level.

---

## Training Parameters

All HuggingFace `TrainingArguments` are externalized in `params.yaml`:

```yaml
TrainingArguments:
  num_train_epochs: 1
  warmup_steps: 500
  per_device_train_batch_size: 1
  weight_decay: 0.01
  logging_steps: 10
  eval_strategy: steps
  eval_steps: 500
  save_steps: 1000000
  gradient_accumulation_steps: 16
```

To change any training hyperparameter, edit `params.yaml` only — no source code changes needed.

---

## API Endpoints

Served by `app.py` via FastAPI on `http://0.0.0.0:8080`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Redirects to `/docs` (Swagger UI) |
| `GET` | `/train` | Triggers full pipeline via `python main.py` |
| `POST` | `/predict` | Returns a summary for the provided `text` |

### Prediction example

```bash
curl -X POST "http://localhost:8080/predict?text=Hannah%3A+Hey%2C+are+you+coming+tonight%3F"
```

The prediction pipeline (`predicition_pipeline.py`) loads the fine-tuned model from `artifacts/model_trainer/` and runs HuggingFace's `pipeline("summarization")` with beam search (`num_beams=8`, `length_penalty=0.8`, `max_length=128`).

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- `conda` or `venv`

### 1. Clone and create environment

```bash
git clone <repo-url>
cd textsummarizer

python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify setup

```bash
python3 -c "from src.textSummarizer.utils.common import read_yaml; print('OK')"
```

---

## Running the Pipeline

Run all 4 stages sequentially:

```bash
python3 main.py
```

Each stage logs its start/completion to both stdout and `logs/continuous_logs.log`. Stages are independent — if you've already run ingestion, the download is skipped automatically.

### Running individual stages via notebooks

The `research/` notebooks mirror each pipeline stage and are designed for iterative development:

| Notebook | Stage |
|----------|-------|
| `1_data_ingestion.ipynb` | Download & extract dataset |
| `2_data_transformation.ipynb` | Tokenize with PEGASUS tokenizer |
| `3_model_trainer.ipynb` | Fine-tune PEGASUS |
| `4_model_evaluation.ipynb` | Compute ROUGE scores |

---

## Running the API

```bash
python3 app.py
```

Then open `http://localhost:8080/docs` for the interactive Swagger UI.

To trigger training via the API:

```bash
curl http://localhost:8080/train
```

---

## Docker

```bash
# Build
docker build -t text-summarizer .

# Run
docker run -p 8080:8080 text-summarizer
```

---

## Key Design Decisions

**Config-driven architecture** — `config/config.yaml` and `params.yaml` are the single source of truth for all paths and hyperparameters. Components never hardcode paths.

**Typed dataclasses as config contracts** — Each pipeline stage receives a strongly-typed `@dataclass` config object (defined in `entity/__init__.py`) rather than a raw dict, catching misconfiguration early.

**`ConfigBox` for dot-access YAML** — `python-box`'s `ConfigBox` wraps the parsed YAML so configs are accessed as `config.model_trainer.root_dir` rather than `config["model_trainer"]["root_dir"]`.

**`@ensure_annotations` runtime type checking** — Applied to utility functions (`read_yaml`, `create_directories`) to enforce type correctness at runtime, catching errors like passing a string where a list is expected.

**Idempotent data ingestion** — `download_file()` checks for the existence of the local zip before downloading, making re-runs safe.

**Gradient accumulation for memory efficiency** — `gradient_accumulation_steps=16` with `per_device_train_batch_size=1` simulates an effective batch size of 16 without requiring a large GPU.

**Structured logging** — All log output is written simultaneously to `logs/continuous_logs.log` and stdout using Python's `logging` module with a consistent `[timestamp: level: module: message]` format.
