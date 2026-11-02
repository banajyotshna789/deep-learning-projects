# Image Caption Generation using CNN + LSTM

> **Note:** Trained model weights and dataset files are not included in this repository because of their size and licensing. To run the training and evaluation scripts, download the dataset first and retrain the models.

## Overview
This project implements an image captioning model using a CNN-RNN architecture. A pretrained ResNet50 model is used as an encoder to extract image features, and an LSTM network is used as a decoder to generate captions.

The model takes an image as input and produces a natural language description.

---

## File Structure

- train.py → training script  
- generate_caption.py → caption generation + visualization  
- evaluate_bleu.py → BLEU score evaluation  
- encoder.py → CNN encoder (ResNet50)  
- decoder_lstm.py → LSTM decoder  
- utils.py → dataset and dataloader  
- vocab.py → vocabulary building  
- run.sh → SLURM training script  
- encoder.pth, decoder.pth → trained model weights  
- results/ → generated outputs  

---

## Requirements

- Python 3.x  
- PyTorch  
- torchvision  
- matplotlib  
- Pillow (PIL)  
- nltk  

Install dependencies:

pip install torch torchvision matplotlib pillow nltk

(Optional, first time only):
python -c "import nltk; nltk.download('punkt')"

---

## Dataset

Dataset used: Flickr8k

Place the dataset in the following structure:

data/
├── Flicker8k_Dataset/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Flickr8k.token.txt

---

## How to Run

### 1. Train the Model

python -u train.py

OR (on GPU cluster):

sbatch run.sh

---

### 2. Generate Captions + Visualization

python generate_caption.py

This will generate:
- captions
- reference vs generated comparison
- images with captions

Saved in:

results/

---

### 3. Evaluate using BLEU Score

python evaluate_bleu.py

Output file:

results/bleu_results.txt

---

## Outputs

All outputs are stored in:

results/

Contents include:
- results.txt → generated captions  
- bleu_results.txt → BLEU scores  
- .png files → image + reference + generated caption  

---

## Model Description

- Encoder: Pretrained ResNet50  
- Decoder: LSTM  
- Loss Function: CrossEntropyLoss  
- Optimizer: Adam  

---

## Evaluation Results

- BLEU-1: 0.6791  
- BLEU-2: 0.4293  
- BLEU-3: 0.2781  
- BLEU-4: 0.1838  

These results show that the model captures key words effectively but struggles with sentence fluency and repetition.

---

## Known Limitations

- Repetition of words (e.g., "dogs dogs", "a a")  
- Incorrect object associations  
- Presence of <unk> tokens  
- Limited grammatical fluency  
- No attention mechanism  

---

## Notes

- Vocabulary tokens are lowercase: <pad>, <start>, <end>, <unk>  
- Decoder hidden size = 256 (must match training and inference)  
- Run all scripts from project root  

---

## Summary

This project demonstrates a complete deep learning pipeline for image captioning using CNN and LSTM. The model successfully learns object-level descriptions but shows limitations in generating fluent captions.
