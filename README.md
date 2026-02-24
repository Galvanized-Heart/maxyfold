<div align="center">

# MaxyFold

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
<!-- These tags are for publications
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)
-->

</div>

## Description

What it does

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
git clone https://github.com/Galvanized-Heart/protein_diff
cd protein_diff

# Create virtual environment for python version 3.11
uv venv --python 3.11

# Sync uv .venv with uv.lock
uv sync --locked
```

## How to run pipeline

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
