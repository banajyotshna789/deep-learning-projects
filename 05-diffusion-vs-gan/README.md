# Comparison of Conditional Diffusion Model and Conditional WGAN-GP for MNIST Image Generation

> **Note:** Trained model weights and dataset files are not included in this repository because of their size and licensing. To run the training and evaluation scripts, download the dataset first and retrain the models.

## 📌 Overview
This project implements and compares two generative models for handwritten digit generation using the MNIST dataset:

- Conditional Denoising Diffusion Probabilistic Model (DDPM)
- Conditional Wasserstein GAN with Gradient Penalty (WGAN-GP)

Both models are trained in a conditional setting, enabling controlled generation of digits (0–9). The goal is to evaluate their performance in terms of image quality, diversity, and training stability.

---

## 📊 Dataset
- MNIST dataset (60,000 training images)
- Image size: 28 × 28 grayscale

---

## ⚙️ Training Details

### Diffusion Model
- Batch size: 64
- Epochs: 30
- Loss: Mean Squared Error (MSE)
- Architecture: U-Net with timestep and label embeddings
- Observation: Stable convergence around epoch 25–30

### Conditional WGAN-GP Model
- Batch size: 64
- Latent dimension: 100
- Learning rate: 1e-4
- Epochs: 50
- Critic steps: 5
- Gradient penalty (λ): 10
- Architecture: Conditional Generator and Critic
- Observation: Improved image quality with training, but some instability remains

---

## 🔍 Results and Observations

- The diffusion model produces smooth, consistent, and diverse digit samples.
- The WGAN-GP model generates sharper digits with better contrast.
- However, the GAN occasionally produces distorted or ambiguous digits.
- Training the GAN for more epochs improves quality but does not fully eliminate instability.

---

## ⚖️ Comparison

| Aspect | Diffusion Model | WGAN-GP |
|--------|----------------|---------|
| Image Quality | Smooth | Sharp |
| Diversity | High | Moderate |
| Training Stability | Stable | Less Stable |
| Generation Speed | Slow | Fast |

---

## 🧠 Key Insight
Diffusion models generate images through a gradual denoising process, resulting in stable training and consistent outputs. In contrast, GANs rely on adversarial learning, which produces sharper images but is more sensitive to training dynamics and hyperparameters.

---

## 📁 Project Structure
gan/                  # Generator and Discriminator models
train_cwgan.py        # GAN training script
train_cond.py         # Diffusion training script
generate_grid.py      # Script to generate 8×8 grid
models/               # Diffusion model components
utils.py              # Utility functions
README.md             # Project documentation

---

## 🚀 How to Run

### Train GAN
python train_cwgan.py

### Generate 8×8 Grid
python generate_grid.py

---

## 🏁 Conclusion
This project highlights a key trade-off in generative modeling:

- Diffusion models prioritize stability, diversity, and consistency.
- GAN models focus on sharpness and faster generation but may suffer from instability.

---

## 🔮 Future Work
- Improve GAN architecture for better stability
- Train for more epochs with fine-tuning
- Evaluate models using quantitative metrics (e.g., FID)
