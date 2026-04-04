# DNN-Clustering

**HDBSCAN** clustering of **adversarial examples** in a deep network’s **feature space**, computation of paper-style **clustering intensity λ**, and **semantics-preserving mutations** seeded from clusters to study defect-detection-style metrics (RQ1 / RQ2 experiments).

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` pins a **CUDA 13.0 nightly** PyTorch build. If that does not match your GPU/driver, install compatible `torch` / `torchvision` wheels from [pytorch.org](https://pytorch.org).

## Prerequisites

- Place trained weights under `./checkpoints/` as `{dataset}_{model}.pth` (must contain `model_state_dict`).  
  Examples: `mnist_lenet1.pth`, `cifar10_resnet20.pth`.
- **ResNet50 (`cifar10_resnet50.pth`) is not included in this repository** because the checkpoint file is too large for typical hosting limits. **Train CIFAR-10 ResNet50 locally, verify accuracy (and any other checks you need), then save** `model_state_dict` in the format expected above.
- Datasets download to `./data` by default (MNIST, CIFAR-10).

## Running experiments

**Experiment 1 (RQ1: clustering and λ)**

```bash
python experiment_1.py
# Single dataset–model pair:
python experiment_1.py --dataset mnist --model lenet1 --device cuda
# Optional: --num_adversarial_samples, --hdbscan_min_cluster_size,
#           --hdbscan_min_samples, --force_regenerate
```

Without `--dataset` / `--model`, all configured pairs are run and results are consolidated.

**Experiment 2 (RQ2: defect-detection potential)**

```bash
python experiment_2.py
python experiment_2.py --dataset cifar10 --model resnet50 --device cuda
# Optional: --n_seeds_per_group, --n_mutations_per_seed
```

Requires artifacts from Experiment 1 (e.g. adversarial libraries). Full sweeps write `./results/experiment_2.json`.

## Main files

| File | Role |
|------|------|
| `experiment_1.py` | Adversarial library → features → HDBSCAN → λ and clustering metrics |
| `experiment_2.py` | Cluster-based seeding + `MutationEngine` + statistical comparison |
| `clustering_analysis.py` | Feature extraction and HDBSCAN |
| `adversarial_generator.py` | FGSM / PGD / DeepFool / AutoAttack (ART) |
| `models.py` | LeNet1/4/5, ResNet20/50 |
| `config.py` | Hyperparameters and paths |

**Utilities** (`tools/`): `hdbscan_param_search.py`, `baseline_lambda_analysis.py`, `npz_viewer.py`. Root: `Coverage Analysis.py`.

## Outputs

- Adversarial library: `./results/{dataset}/{model}/adversarial_library.npz`
- Experiment 1: `./results/{dataset}/{model}/experiment_1/` (and consolidated JSON under `./results/`)
- Sample bundled results: `results/` in the repo

## License

Follow any additional terms required by the original paper, datasets, or upstream code.
