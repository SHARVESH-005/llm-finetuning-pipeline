# LLM Fine-Tuning and Alignment Pipeline

This repository contains a production-grade pipeline for fine-tuning and aligning Large Language Models (LLMs). It mirrors the workflows used at top AI organizations, covering data preparation, Supervised Fine-Tuning (SFT) via QLoRA, Direct Preference Optimization (DPO), and high-performance serving via vLLM.

## Architecture

1. **Data Curation**: Scripts to ingest, clean, and format HuggingFace datasets for SFT and DPO formats.
2. **SFT (QLoRA)**: Uses `peft` and `bitsandbytes` to efficiently fine-tune the model in 4-bit precision.
3. **Alignment (DPO)**: Uses the `trl` library to align the model against preference datasets, acting as a modern replacement for RLHF.
4. **Serving**: Uses `vLLM` and FastAPI for high-throughput, PagedAttention-backed API serving.
5. **Evaluation**: Integrates with `lm-evaluation-harness` to mathematically prove model improvements across standard benchmarks.

## Setup Instructions

### 1. Local Environment
If you have a local GPU (e.g., RTX 3090/4090):
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Free Cloud Setup (Google Colab)
If you do not have a local GPU, this pipeline is designed to run on a free Google Colab T4 instance using the TinyLlama model. 
- Open the provided `run_on_colab.ipynb` file in Google Colab.
- It will automatically set up the environment and execute the pipeline steps.

## Running the Pipeline

All configuration parameters (model name, batch sizes, LoRA settings) are centralized in `configs/pipeline_config.yaml`.

### Step 1: Data Preparation
```bash
python src/data/prepare_dataset.py
```

### Step 2: Supervised Fine-Tuning (SFT)
```bash
python src/training/sft_trainer.py
```

### Step 3: Direct Preference Optimization (DPO)
```bash
python src/training/dpo_trainer.py
```

### Step 4: Serving (vLLM)
```bash
python src/serving/vllm_server.py
```
*(Note: vLLM is generally designed for Linux/WSL. If running on Windows locally, it may require WSL2).*

### Step 5: Evaluation
```bash
python src/evaluation/run_eval.py
```
