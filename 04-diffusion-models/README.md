# Text-Conditioned Diffusion Model for Handwritten Digit Generation

> **Note:** Trained model weights and dataset files are not included in this repository because of their size and licensing. To run the training and evaluation scripts, download the dataset first and retrain the models.

## 📌 Overview
This project implements a **Denoising Diffusion Probabilistic Model (DDPM)** to generate handwritten digit images using the MNIST dataset.

Two models are developed:
- **Unconditional Diffusion Model (Baseline)**
- **Text-Conditioned Diffusion Model**, which generates digits based on labels ("zero" to "nine")

The project demonstrates how conditioning improves control and interpretability in image generation.

---

## 🧠 Model Description

### Diffusion Model (DDPM)
The model follows the DDPM framework:
- A **forward process** gradually adds Gaussian noise to images
- A **reverse process** learns to remove noise step-by-step
- Training objective: Mean Squared Error (MSE) between predicted and true noise

### Architecture
- Backbone: **U-Net**
- Downsampling and upsampling layers with skip connections
- Time embedding: **Sinusoidal positional encoding**
- Noise schedule: **Cosine beta schedule**

### Conditional Model
- Uses **learned embeddings** for digit labels (0–9)
- Combines label embedding with time embedding
- Injects conditioning into the U-Net
- Enables controlled digit generation

---

## 📂 Project Structure

diffusion_project/
├── data/                      # MNIST dataset
├── outputs/                   # Generated images & checkpoints
├── diffusion_env/             # Python virtual environment
├── model.py                   # Baseline DDPM model
├── cond_model.py              # Conditional DDPM model
├── train.py                   # Train baseline model
├── train_cond.py              # Train conditional model
├── sample.py                  # Sample baseline model
├── sample_cond.py             # Sample conditional model
├── utils.py                   # Utility functions
├── README.md                  # Project documentation

---

## ⚙️ Setup Instructions (IU HPC)

1. Request GPU  
srun -A <your-account> -p gpu --gres=gpu:1 --time=04:00:00 --pty bash

2. Navigate to project  
cd <project-root>/diffusion_project

3. Load Python module  
module load python/gpu/3.10.10

4. Activate virtual environment  
source diffusion_env/bin/activate

5. Install dependencies (if needed)  
pip install torch torchvision torchaudio

---

## 🚀 How to Run

Train Unconditional Model  
python train.py

Train Conditional Model  
python train_cond.py

Generate Samples (Unconditional)  
python sample.py

Generate Samples (Conditional)  
python sample_cond.py

---

## 📊 Results

### Unconditional Model
- Generates random digit-like images  
- No control over digit class  
- Output: outputs/final_samples.png  

### Conditional Model
- Generates digits corresponding to labels (0–9)  
- Produces structured and interpretable outputs  
- Outputs:  
  outputs/final_cond_samples.png  
  outputs/final_cond_8x8.png  

---

## 📈 Training Details

- Dataset: MNIST (60,000 training images)  
- Image size: 28 × 28  
- Batch size: 64 / 128  
- Epochs: 30  
- Final loss: ~0.0363  
- Convergence observed around epoch 25–30  

---

## 🔍 Observations

- Conditional model produces significantly more structured outputs  
- Digits align well with intended labels (0–9)  
- 8×8 grid demonstrates both consistency and diversity  
- Unconditional model produces noisy and less interpretable images  
- Some digits (e.g., 4 and 5) are slightly ambiguous  
- Image sharpness is limited due to model size and training duration  

---

## ⚠️ Limitations

- Generated images are not perfectly sharp  
- Small U-Net limits model capacity  
- No advanced conditioning techniques (e.g., classifier-free guidance)  
- Diffusion models typically require longer training for high-quality outputs  

---

## 💡 Future Improvements

- Train for more epochs  
- Increase model capacity  
- Use classifier-free guidance  
- Reduce diffusion timesteps for faster sampling  
- Apply the model to more complex datasets  

---

## 🏆 Summary

This project demonstrates:
- Implementation of DDPM from scratch  
- Extension to conditional diffusion  
- Controlled image generation using label embeddings  
- Clear comparison between unconditional and conditional models  


