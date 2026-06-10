# Usage Guide

This document describes the expected workflow for reproducing the code experiments in the manuscript.

## 1. Prepare the Environment

Install Python dependencies from the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install PyTorch and CUDA versions that match the local hardware. The versions in `requirements.txt` document the environment used for this release.

## 2. Prepare Data

Place seismic data, horizon labels, and fault labels under `data/`. The scripts assume local data files and do not download datasets automatically.

The data preparation scripts are:

- `data/make_dataset_labels.py` - builds structural-framework labels from interpreted horizons and faults.
- `data/make_datasets.py` - creates train/validation/test splits.
- `data/data_aug.py` - applies data augmentation for model training.

Run these scripts only after the required raw files are available in the expected local paths.

## 3. Run the Quick Test

Before long training jobs, run:

```bash
python tests/quick_test.py
```

The quick test verifies that the repository is complete enough to run as source code. It checks required files, compiles Python files, and validates the dependency list for duplicate package names.

## 4. Train Models

Single-task training:

```bash
python demo_classification.py
```

Joint horizon/fault training:

```bash
python demo_classification_two_tasks.py
```

The scripts define command-line arguments internally with `argparse`. Use `python demo_classification.py --help` or `python demo_classification_two_tasks.py --help` to inspect available options.

## 5. Evaluate Models

Single-task evaluation:

```bash
python evaluate_classification.py
```

Joint-task evaluation:

```bash
python evaluate_classification_two_tasks.py
```

Ensure that model checkpoints and the corresponding test data are available at the paths configured in the scripts or command-line arguments.

## 6. Post-process and Visualize

Separate structural-framework predictions into fault and horizon components:

```bash
python separation_fault_and_horizon.py
```

Visualize DINOv2 feature maps:

```bash
python plt_tezhengtu.py
```

Visualize seismic gradient/RGT features:

```bash
python plt_gradient_RGT.py
```
