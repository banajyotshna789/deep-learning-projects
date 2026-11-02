# Deep Learning Projects

PyTorch projects covering image segmentation, image captioning, diffusion models, and GANs, completed for the graduate Deep Learning course (CSCI-P 558) at Indiana University in Spring 2026.

| # | Project | Methods | Result |
|---|---|---|---|
| 1 | [Image segmentation](01-image-segmentation/) | U-Net with ResNet18 and ResNet34 encoders, Oxford-IIIT Pet | ResNet34: Dice 0.945, IoU 0.896 |
| 2 | [Image captioning with an LSTM](02-image-captioning-lstm/) | ResNet50 encoder, LSTM decoder, Flickr8k | BLEU-1 0.679, BLEU-4 0.184 |
| 3 | [Image captioning with a Transformer](03-image-captioning-transformer/) | ResNet50 encoder, Transformer decoder, Flickr8k | BLEU-1 0.640, BLEU-4 0.246 |
| 4 | [Diffusion models](04-diffusion-models/) | Unconditional and label-conditioned DDPM with a U-Net, MNIST | [Generated digit samples](04-diffusion-models/final_outputs/) |
| 5 | [Diffusion vs. GAN](05-diffusion-vs-gan/) | Conditional DDPM and conditional WGAN-GP, MNIST | Qualitative comparison of quality, diversity, and stability |
| 6 | [World models and agentic AI](06-world-models-agentic-ai-review/) | Literature review paper | [Paper (PDF)](06-world-models-agentic-ai-review/report.pdf) |

Results are the values recorded in each project's README or report when the project was completed. They weren't rerun for this repository.

## Final project

My team's final project for this course, [Multimodal Image Captioning](https://github.com/banajyotshna789/multimodal-image-captioning), has its own repository.

## Running the code

Each project folder has its own README with setup and run instructions. Trained weights and datasets are not included because of their size and licensing, so download the dataset and retrain the models before running the evaluation scripts. Training was done on GPUs on an HPC cluster, so replace placeholders such as `<your-account>` and `<project-root>` in the scripts with your own values.
