# Multimodal Image Captioning Project

> **Note:** Trained model weights and dataset files are not included in this repository because of their size and licensing. To run the training and evaluation scripts, download the dataset first and retrain the models.

## Overview
This project implements a complete multimodal pipeline consisting of:

- Generative Models (LSTM + Transformer)
- Discriminative Model (image-caption matching)
- Fusion Models (baseline, concat, gated)

The system generates captions, evaluates them, and improves results using fusion and ranking.

---

## Project Structure

### Generative Models (Member 1: Banajyotshna Devi)
- train.py
- train_transformer.py
- generate_captions_v1.py
- inference.py
- preprocess_data.py
- compute_all_metrics.py

### Discriminative Model (Banajyotshna Devi)
- discriminative_model.py
- discriminative_dataset.py
- train_discriminative.py
- infer_discriminative.py

### Fusion Model (Member 3)
member3_fusion/
- fusion_dataset.py
- fusion_models.py
- train_fusion.py
- plot_auc_graphs.py
- checkpoints/
- auc_graphs.png

---

## Dataset
- train.csv
- test.csv
- clean_data.csv
- discriminative_data.csv
- vocab.pkl

---

## Models
- caption_model_v1.pth
- transformer_model_v1.pth
- discriminative_model_v1.pth
- discriminative_model_v2.pth

---

## Results
- results_input_v1.csv
- results_input_v2.csv
- discriminative_results.csv
- discriminative_reports/
- compare_*.png

---

## Requirements

Install dependencies:

pip install torch torchvision numpy pandas Pillow scikit-learn matplotlib nltk

---

## How to Run

### Generative Models
python generate_captions_v1.py
python compute_all_metrics.py

### Discriminative Model
python infer_discriminative.py

### Fusion Models
cd member3_fusion
python train_fusion.py --model baseline
python train_fusion.py --model concat
python train_fusion.py --model gated

### Plot AUC Graph
python plot_auc_graphs.py

---

## Evaluation Metrics

### Generative
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- ROUGE-L

### Fusion / Discriminative
- Accuracy
- Precision
- Recall
- F1 Score
- AUC (ROC Curve)

---

## Final Results

### Generative Model
- BLEU-1: 0.4000
- BLEU-2: 0.2488
- BLEU-3: 0.1498
- BLEU-4: 0.1239
- ROUGE-L: 0.4876

### Fusion Model

| Model | Accuracy | F1 | AUC |
|------|---------|----|-----|
| Baseline | ~0.47 | ~0.00 | ~0.45 |
| Concat | ~0.75 | ~0.80 | ~0.82 |
| Gated | ~0.81 | ~0.82 | ~0.87 |

---

## Key Insights

- Transformer generates more detailed captions than LSTM
- Discriminative model improves caption ranking
- Fusion significantly improves matching performance
- Gated fusion performs best among all methods
- Dataset bias limits generalization

---

## Conclusion

This project demonstrates a complete multimodal system combining:

- Caption generation
- Caption evaluation
- Feature fusion

The fusion model improves performance by effectively combining image and text features.

