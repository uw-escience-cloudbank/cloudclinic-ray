<div align="center">

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![uv](https://img.shields.io/badge/uv-package%20manager-4051b5.svg)](https://docs.astral.sh/uv/)
[![Ray](https://img.shields.io/badge/Ray-028CF0?style=flat&logo=ray&logoColor=white)](https://docs.ray.io/en/latest/)
[![SkyPilot](https://img.shields.io/badge/SkyPilot-372F8A?style=flat)](https://docs.skypilot.co/en/stable/)

# cloudclinic-ray

</div>

[Ray](https://docs.ray.io/en/latest/) is an open-source framework for scaling AI
and Python applications. It provides general-purpose distributed computing primitives
and specialized libraries for common machine learning tasks. This repository contains
two examples using different parts of Ray:

1. **Distributed NLP analysis ([core.py](src/cloudclinic_ray/core.py)):**
   Uses [Ray Core](https://docs.ray.io/en/latest/ray-core/walkthrough.html) actors to count passive linguistic
   constructions in the [Brown corpus](https://www.nltk.org/nltk_data/).
2. **Distributed image classifier ([train.py](src/cloudclinic_ray/train.py)):**
   Uses [Ray Train](https://docs.ray.io/en/latest/train/train.html) to train a PyTorch classifier
   on [Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) across multiple workers.

Both demos can be run locally or on cloud clusters provisioned via
[SkyPilot](https://docs.skypilot.ai/en/latest/docs/index.html).

> [!NOTE]
> If you're new to SkyPilot, refer to this
> [CloudBank Cloud Clinic](https://www.youtube.com/watch?v=S1Zm7JX6qEk), its
> [slides](https://bit.ly/4bWeE7b), and the corresponding
> [demo project](https://github.com/uw-escience-cloudbank/skypilot-demo) for a quick introduction.

## Quickstart

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone this
repository, and open a terminal at the project root.

### Set up the environment

Install the dependencies and activate the environment:

```bash
uv sync --frozen --all-extras
source .venv/bin/activate
```

### Configure cloud access

Log in to your SkyPilot API server:

```bash
sky api login
sky api info
```

SkyPilot saves the connection for subsequent `sky` commands.

## Demo 1 ([core.py](src/cloudclinic_ray/core.py))

Use Ray actors to analyze the Brown corpus and count passive constructions by
document genre.

Run locally using the `core-demo` command:

```text
usage: core-demo [-h] [--n-workers N_WORKERS]

options:
  -h, --help            show this help message and exit
  --n-workers N_WORKERS Number of workers
```

Use the `sky` command to launch this demo on cloud resources:

```bash
sky launch -c ray-core sky/ray_core.yaml
```

## Demo 2 ([train.py](src/cloudclinic_ray/train.py))

Use Ray Train and PyTorch to classify Fashion-MNIST images. The program reports
test loss and accuracy after each epoch.

Run locally using the `train-demo` command:

```text
usage: train-demo [-h] [--n-workers N_WORKERS] [--gpu] [--epochs EPOCHS]

options:
  -h, --help            show this help message and exit
  --n-workers N_WORKERS Number of workers
  --gpu                 Require one GPU per worker
  --epochs EPOCHS       Number of training epochs
```

Use the `sky` command to launch this demo on cloud resources:

```bash
sky launch -c ray-train sky/ray_train.yaml
```

## Manage cloud runs

Check the cluster status or terminate it once it has finished:

```bash
sky status
sky down ray-core
sky down ray-train
```

## Additional resources

- Overview [Ray Core](https://docs.ray.io/en/latest/ray-core/key-concepts.html) / [Ray Data](https://docs.ray.io/en/latest/data/key-concepts.html) / [Ray Train](https://docs.ray.io/en/latest/train/overview.html)
- [Ray Use Cases](https://docs.ray.io/en/latest/ray-overview/use-cases.html)
- [Ray Training with SkyPilot](https://docs.skypilot.ai/en/stable/examples/training/ray.html)
