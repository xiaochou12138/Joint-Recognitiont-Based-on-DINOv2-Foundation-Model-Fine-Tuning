# Joint Recognition Based on DINOv2 Foundation Model Fine-Tuning

This repository contains the source code accompanying the manuscript
**"Joint Recognition of Seismic Horizons and Faults Based on DINOv2 Foundation Model Fine-Tuning"**.

The code fine-tunes DINOv2-style vision foundation backbones for seismic interpretation tasks, including horizon recognition, fault recognition, and structural-framework prediction.

## Repository Contents

- `demo_classification.py` - single-task training entry point.
- `demo_classification_two_tasks.py` - multi-task training entry point for joint seismic interpretation.
- `evaluate_classification.py` - evaluation script for single-task predictions.
- `evaluate_classification_two_tasks.py` - evaluation script for multi-task predictions.
- `dataset.py` - dataset loader used by the training and evaluation scripts.
- `models/` - DINOv2 backbones, adapters, DPT modules, and U-Net baseline components.
- `loss/` - Dice, weighted Dice, focal, SSIM, and metric utilities.
- `data/` - scripts for label construction, dataset generation, and data augmentation.
- `run/` - example shell commands for model training.
- `tests/quick_test.py` - lightweight repository smoke test.
- `docs/` - usage notes and manuscript-ready code availability text.

## Installation

Create a clean Python environment and install the dependencies:

```bash
git clone https://github.com/xiaochou12138/Joint-Recognitiont-Based-on-DINOv2-Foundation-Model-Fine-Tuning.git
cd Joint-Recognitiont-Based-on-DINOv2-Foundation-Model-Fine-Tuning
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The full training workflow requires PyTorch-compatible hardware. GPU acceleration is recommended for DINOv2 fine-tuning.

## Data

The training and evaluation scripts expect seismic volumes, horizon labels, and fault labels arranged under `data/`. The manuscript experiments use synthetic seismic records and the F3 dataset.

Large research data files are not stored directly in this repository. Download the data from the project data link and place the extracted files under `data/` before running the training scripts:

<https://drive.google.com/drive/folders/1gKgE0lJfkmGsF433mj0jMVanPNZjeJKP?usp=drive_link>

Recommended layout:

```text
data/
  synthetic_data/
    train/
    valid/
    test/
  F3/
```

## Quick Test

Run the lightweight smoke test before training:

```bash
python tests/quick_test.py
```

This test checks that the repository contains the expected source files, that Python files compile, and that `requirements.txt` does not contain duplicate package entries. It does not download data or train a model.

## Runnable Example with Data and Trained Models

The `example/` folder contains a small synthetic dataset and trained checkpoints for the structural-framework recognition task. To run inference with the included model:

```bash
python example/test.py
```

For CPU-only execution:

```bash
python example/test.py --device cpu
```

See [example/README.md](example/README.md) for the example data layout, checkpoint descriptions, inference commands, and training commands.

## Training and Evaluation

Single-task training:

```bash
python demo_classification.py
```

Multi-task joint training:

```bash
python demo_classification_two_tasks.py
```

Single-task evaluation:

```bash
python evaluate_classification.py
```

Multi-task evaluation:

```bash
python evaluate_classification_two_tasks.py
```

Post-processing for separating structural-framework predictions into horizon and fault components:

```bash
python separation_fault_and_horizon.py
```

Feature-map visualization:

```bash
python plt_tezhengtu.py
```

More detailed usage notes are available in [docs/USAGE.md](docs/USAGE.md).

## Computer Code Availability

The code is publicly available from this GitHub repository:

<https://github.com/xiaochou12138/Joint-Recognitiont-Based-on-DINOv2-Foundation-Model-Fine-Tuning>

It can be downloaded anonymously through the GitHub web interface or with `git clone`. A manuscript-ready code availability statement is provided in [docs/COMPUTER_CODE_AVAILABILITY.md](docs/COMPUTER_CODE_AVAILABILITY.md).

## License

This repository is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.
