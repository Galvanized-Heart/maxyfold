<div align="center">

# MaxyFold

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
<!-- These tags are for publications
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)
-->
</div>

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Data Pipeline](#data-pipeline)
  - [1. Download filtered PDB data](#1-download-filtered-pdb-data)
  - [2. Process to clean LMDB backend](#2-process-to-clean-lmdb-backend)
  - [3. Create manifest](#3-create-manifest)
  - [4. Generate train/val/test splits](#4-generate-trainvaltest-splits)
- [Training & Inference](#training--inference)
- [Project Status & Roadmap](#project-status--roadmap)

## Description
MaxyFold is an open source PyTorch Lightning reimplementation of **all-atom diffusion models** for protein structure prediction (inspired by AlphaFold3, Boltz-1, Chai-1 architectures) from scratch.

MaxyFold focuses on building a production ready with a **data + training pipeline** rather than pre-trained weights. It's an excellent solution for researchers/engineers who want full control over data curation, filtering, and extensibility in AI for structural biology. 

Key strengths:
- **Dynamic PDB bulk download** with filters (atomic resolution, method, date cutoff, etc.).
- **Modular DataBackend** (LMDB default. Easy to subclass for HDF5 or other custom stores).
- **Principled splitting** via MMseqs2 + Scaffold clustering to avoid data leakage (sequence + substrate aware).
- End-to-end scalable pipeline using Hydra configs, PyTorch Lightning, uv dependency management, pytest suite.

If you're working on diffusion for biomolecules and need a clean foundation for experimentation, this repo is a great place to get started.

## Installation

This project uses [astral uv](https://docs.astral.sh/uv/) for dependency management.

If you haven't already installed uv, you can run this:

```bash
# Download and install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Download the repo and sync the environment.

```bash
# Clone repository
git clone https://github.com/Galvanized-Heart/maxyfold
cd maxyfold

# Set project root
export PROJECT_ROOT=$(pwd)

# Create virtual environment for python version 3.11
uv venv --python 3.11

# Sync uv .venv with uv.lock
uv sync --locked
```

If you don't already have MMseqs2 installed you can run this:
```bash
# Download and install mmseqs2
wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
tar xvfz mmseqs-linux-avx2.tar.gz
export PATH=$(pwd)/mmseqs/bin/:$PATH

# Move to .local/bin and delete install files
mv mmseqs/bin/mmseqs ~/.local/bin/
rm -rf mmseqs mmseqs-linux-avx2.tar.gz
```

## Data Pipeline
### 1. Download filtered PDB data
```bash
maxyfold download --ids --assemblies --ccd --batch-size 10000 --file-limit 5000
```
(Custom filters build upon `configs/query/pdb.yaml`)

### 2. Process to clean LMDB backend
```bash
maxyfold process --file-limit 1000
```

### 3. Create manifest (sequences, ligands, etc.)
```bash
maxyfold manifest
```

### 4. Generate train/val/test splits (MMseqs2)
```bash
maxyfold split --seq-id 0.3 --coverage 0.8 --cluster-mode 1
```

Want to use a different backend? Subclass DataBackend (see src/maxyfold/data/storage/lmdb.py for example) and plug it into your config.

## Training/Inference

Train model with default configurations.

```bash
# Train on CPU
uv run src/train.py trainer=cpu

# Train on GPU
uv run src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/).

```bash
uv run src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this:

```bash
uv run src/train.py trainer.max_epochs=20 data.batch_size=64
```

## How to run tests

All pytest functionality is the same as seen in the [pytest documentation](https://docs.pytest.org/en/stable/).

```bash
uv run pytest [OPTIONS]
```

## Project Status & Roadmap
- ✅ Fully configurable data download + processing pipeline.
- ✅ LMDB backend with safetensors serialization.
- ✅ MMseqs2-based splitting to prevent leakage.
- 🔄 SE(3)-equivariant Transformer backbone in progress.
- 🔜 Sample generation + evaluation metrics (RMSD, GDT, etc.).
- 🔜 Pre-trained checkpoints & Gradio/Streamlit demo.