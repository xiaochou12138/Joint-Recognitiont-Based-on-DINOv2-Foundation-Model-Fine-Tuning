# Example: Data and Trained Models

This folder contains a runnable example for the structural-framework recognition task. It includes a small synthetic seismic dataset, trained model checkpoints, and scripts for training and inference.

## Contents

- `data/synthetic_data/` - example seismic data and labels in `.dat` format.
- `checkpoint_structure_task/` - trained checkpoints for the structure task.
- `train.py` - example training script.
- `test.py` - example inference and visualization script.
- `dataset.py` - dataset loader used by the example scripts.
- `log/` - text logs from example training runs.

The example uses relative paths inside this directory. The scripts automatically add the repository root to `PYTHONPATH`, so they can import `models/` and `loss/` from the main project.

## Included Data

The included dataset is arranged as:

```text
data/synthetic_data/
  train/
    seis1/   # training seismic inputs
    label1/  # training structural-framework labels
  valid/
    seis/    # validation seismic inputs
    label/   # validation structural-framework labels
```

Each `.dat` file stores a `224 x 224` `float32` array.

## Included Trained Models

The trained checkpoints are stored under:

```text
checkpoint_structure_task/structure/mse/
  pup/
    lora_small_minloss_valid.pth
    lora_small_minloss_valid_lora.pth
    lora_small_minloss_train.pth
    lora_small_minloss_train_lora.pth
  unet/
    unfrozen_large_minloss_valid.pth
    unfrozen_large_minloss_train.pth
```

The default inference setting in `test.py` uses the DINOv2 PUP model with LoRA fine-tuning:

```text
dataset=structure
loss=mse
netType=pup
checkpointName=lora
vt=small
```

## Run Inference with the Trained Model

From the repository root:

```bash
python example/test.py
```

The script loads the validation sample from `example/data/synthetic_data/valid/`, loads the trained checkpoint from `example/checkpoint_structure_task/`, and displays the seismic input, prediction, and target label.

On a machine without a CUDA GPU, run:

```bash
python example/test.py --device cpu
```

## Run Example Training

From the repository root:

```bash
python example/train.py
```

The default training command uses the included `structure` data and saves checkpoints under `example/checkpoint_structure_task/`.

For CPU-only testing, use:

```bash
python example/train.py --device cpu --epochs 1
```

For the U-Net example checkpoint, use:

```bash
python example/test.py --netType unet --checkpointName unfrozen --vt large --device cpu
```

## Notes

- GPU inference/training is recommended for DINOv2-based models.
- The example checkpoints are included for reproducibility and quick inspection; full manuscript-scale training may require the larger dataset described in the main `README.md`.
- Generated visualization outputs under `png/` and TensorBoard event files are local run artifacts and are not required to use the example.
