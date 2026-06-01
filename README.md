# CV HW4 - Image Restoration with PromptIR

## Introduction

This project addresses the task of blind image restoration, specifically targeting
two types of degradation: rain streaks and snow. The goal is to train a single
unified model that can restore clean images from degraded inputs without prior
knowledge of the degradation type. We adopt **PromptIR** (Potlapalli et al.,
NeurIPS 2023) as our base model, a Transformer-based framework that uses learnable
prompt vectors to encode degradation-specific information and guide the decoder
accordingly. To improve performance, we trained the model on full-resolution
256×256 patches and applied Test Time Augmentation (TTA) at inference time.
Our final model achieves a PSNR of **31.38 dB** on the public leaderboard.

---

## Environment Setup

### 1. Clone PromptIR
```bash
git clone https://github.com/va1shn9v/PromptIR.git
```

### 2. Install dependencies
```bash
pip install torch torchvision
pip install lightning einops
```

### 3. Project structure
```
cv_hw4/
├── PromptIR/          # cloned from GitHub
├── dataset/
│   ├── train/
│   │   ├── degraded/  # rain-1.png ~ rain-1600.png, snow-1.png ~ snow-1600.png
│   │   └── clean/     # rain_clean-1.png ~ ..., snow_clean-1.png ~ ...
│   └── test/
│       └── degraded/  # 0.png ~ 99.png
├── dataset.py
├── train.py
├── infer.py
├── visualize.py
└── plot.py
```

---

## Usage

### Training
```bash
python3 train.py \
    --data_root dataset/train \
    --ckpt_dir checkpoints_p256 \
    --epochs 100 \
    --batch_size 2 \
    --patch_size 256 \
    --gpu 1
```

Checkpoints are saved under `checkpoints_p256/`. The best checkpoint is selected
based on validation PSNR and saved as `best.pth`.

### Inference
```bash
python3 infer.py \
    --test_root dataset/test \
    --ckpt checkpoints_p256/best.pth \
    --output pred.npz \
    --gpu 1 \
    --tta
```

The output `pred.npz` contains 100 restored images stored as `(3, H, W)` uint8
arrays, with filenames as keys (e.g., `0.png`, `1.png`, ...).

### Plot Training Curves
```bash
python3 plot.py \
    --log checkpoints_p256/train_log.csv \
    --output curves.png
```

### Visualization
```bash
python3 visualize.py \
    --ckpt checkpoints_p256/best.pth \
    --data_root dataset/train \
    --output visualization.png \
    --gpu 1
```

---

## Performance Snapshot

The final model achieves a validation PSNR of **30.64 dB** and a public leaderboard
PSNR of **31.38 dB** (with TTA applied at inference time).

![Leaderboard](image.png)
