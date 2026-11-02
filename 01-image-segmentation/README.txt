CSCI-P 558 – Deep Learning
Project 1: Semantic Image Segmentation with U-Net

Author: Banajyotshna Devi
Indiana University
Spring 2026

------------------------------------------------------------

NOTE: Trained model weights are not included in this repository because
of their size. Run the training script to recreate them.

------------------------------------------------------------

PROJECT DESCRIPTION

This project implements semantic image segmentation using the U-Net architecture.
The goal is to segment pets from background using the Oxford-IIIT Pet dataset.

Two encoder backbones are compared:

1. U-Net with ResNet18
2. U-Net with ResNet34

Both models are evaluated using Intersection over Union (IoU) and Dice Score.

------------------------------------------------------------

DATASET

Oxford-IIIT Pet Dataset

- Binary segmentation (Pet vs Background)
- Images resized to 128 x 128
- 80% training / 20% validation split
- Dataset automatically downloaded on first run
- Dataset files are NOT included in this archive

------------------------------------------------------------

TRAINING SETTINGS

- Image size: 128 x 128
- Batch size: 8
- Learning rate: 1e-4
- Optimizer: Adam
- Loss function: Binary Cross-Entropy with Logits
- Epochs: 10
- Hardware used: NVIDIA Tesla V100 GPU

------------------------------------------------------------

HOW TO RUN

Step 1: Load Python module on Quartz

module load python/gpu/3.10.10

Step 2: Train both models

python train_segmentation.py

This script:
- Trains U-Net with ResNet18
- Trains U-Net with ResNet34
- Saves model_resnet18.pth
- Saves model_resnet34.pth
- Prints final performance comparison

------------------------------------------------------------

VISUALIZATION

To generate side-by-side comparison:

python visualize_compare.py

This produces:
comparison_resnet18_vs_resnet34.png

To visualize a single model:

Open visualize.py and change:

backbone = "resnet34"

to either:

backbone = "resnet18"
or
backbone = "resnet34"

Then run:

python visualize.py

This generates:
segmentation_resnet18.png
or
segmentation_resnet34.png

------------------------------------------------------------

FINAL RESULTS

ResNet18
Dice: 0.9389
IoU: 0.8854

ResNet34
Dice: 0.9451
IoU: 0.8963

ResNet34 achieves slightly better performance.

------------------------------------------------------------

PROJECT FILES

train_segmentation.py
visualize.py
visualize_compare.py
model_resnet18.pth
model_resnet34.pth
comparison_resnet18_vs_resnet34.png
segmentation_resnet18.png
segmentation_resnet34.png
README.txt
report.pdf

------------------------------------------------------------

NOTES

- Dataset is not included in the archive.
- Visualization uses the same fixed image for fair comparison.
- Results may vary slightly due to randomness.
