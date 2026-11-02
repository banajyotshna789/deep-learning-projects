# Image Captioning using Transformer (Project 3)

> **Note:** Trained model weights and dataset files are not included in this repository because of their size and licensing. To run the training and evaluation scripts, download the dataset first and retrain the models.

## Overview
This project extends Project 2 by replacing the LSTM decoder with a Transformer-based decoder for image caption generation. The goal is to compare the performance of RNN (LSTM) and Transformer architectures on the Flickr8k dataset.

---

## Files

- train_transformer.py → training script for Transformer
- decoder_transformer.py → Transformer decoder
- generate_caption_transformer.py → caption generation
- compare_models.py → LSTM vs Transformer comparison
- evaluate_bleu_transformer.py → BLEU evaluation
- encoder.py → CNN encoder (ResNet)
- decoder_lstm.py → LSTM baseline (Project 2)
- utils.py → dataset loader
- vocab.py → vocabulary
- run.sh → SLURM script
- encoder_transformer_best.pth → trained encoder
- decoder_transformer_best.pth → trained decoder
- results/ → output images and captions

---

## Requirements

- Python 3.x
- PyTorch
- torchvision
- nltk
- matplotlib
- pillow

Install dependencies:
pip install torch torchvision nltk matplotlib pillow

---

## Dataset

Place Flickr8k dataset in:

data/
├── Flicker8k_Dataset/
└── Flickr8k.token.txt

---

## Training

python -u train_transformer.py

---

## Caption Generation

python generate_caption_transformer.py

---

## Model Comparison

python compare_models.py

---

## BLEU Evaluation

python evaluate_bleu_transformer.py

---

## Results

Transformer model:
- BLEU-1: 0.6403
- BLEU-4: 0.2457

LSTM model:
- BLEU-1: 0.6791
- BLEU-4: 0.1838

---

## Key Observations

- Transformer produces more fluent captions
- Transformer avoids repetition
- LSTM has higher BLEU-1 but worse sentence structure
- Transformer has significantly higher BLEU-4

---

## Summary

The Transformer model improves sentence-level coherence and caption quality compared to the LSTM baseline, demonstrating the advantage of attention-based architectures.
