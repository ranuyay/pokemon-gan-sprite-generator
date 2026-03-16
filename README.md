# Pokémon GAN Sprite Generator

Synthetic Pokémon sprites generated using **Generative Adversarial Networks (GANs)**.

This project investigates how GAN architecture and hyperparameter choices affect **training stability and image quality** when working with a relatively small but visually complex dataset.

The work began as a graduate deep learning assignment and was later refactored into a **modular machine learning repository** for portfolio presentation.

---

## Project Overview

This project implements and compares multiple GAN training approaches for image generation:

* **DCGAN** with Binary Cross-Entropy loss
* **WGAN** with weight clipping
* **WGAN-GP** with gradient penalty

The goal was to understand how **loss functions, model capacity, and regularization** influence GAN training behavior.

Models were trained on a **Pokémon sprite dataset** to generate new synthetic sprites.

---

## Dataset

**Dataset:** Pokémon Sprite Dataset
**Source:** Dive Into Deep Learning (d2l), originally sourced from PokémonDB

Original dataset:

* **40,597 images**
* **721 Pokémon classes**

After filtering classes with fewer than **30 images**:

* **39,285 images**
* **669 classes**

Sprites include variations such as:

* front and back views
* shiny variants
* multiple game generations
* alternate forms

---

## Preprocessing Pipeline

Exploratory data analysis revealed several issues:

* dominant white backgrounds
* inconsistent transparency encoding
* palette-mode images
* variable sprite sizes

To address this, a preprocessing pipeline was built that:

1. convert palette images to RGB
2. composite sprites over a white background
3. detect sprite boundaries using transparency or color thresholding
4. crop to the sprite bounding box
5. pad the image to square dimensions
6. resize to **64 × 64**
7. normalize pixel values to **[-1, 1]**

This preprocessing step significantly improved **GAN training stability**.

---

## GAN Architectures

### DCGAN

Baseline convolutional GAN architecture using:

* transposed convolutions for upsampling
* convolutional discriminator
* binary cross-entropy loss
* Adam optimizer

While it produced recognizable sprites, training proved **unstable**.

---

### WGAN

Replaced BCE loss with **Wasserstein loss** to improve gradient behavior.

Weight clipping was used to enforce the Lipschitz constraint, but this resulted in **mode collapse**, where the generator produced repetitive outputs.

---

### WGAN-GP

Final architecture used **Wasserstein GAN with Gradient Penalty**, which:

* replaces weight clipping
* stabilizes critic gradients
* improves training dynamics

Key hyperparameters explored:

* base filter count (`n`)
* gradient penalty coefficient (`λ`)
* number of critic updates (`n_critic`)

---

## Evaluation

Model performance was evaluated using **Fréchet Inception Distance (FID)**.

FID compares the distribution of generated images to real images using features extracted from an **InceptionV3 network**.

Lower FID values indicate **greater similarity between real and generated image distributions**.

---

## Repository Structure

```
pokemon-gan-sprite-generator/
│
├── data/
├── models/
├── notebooks/
├── report/
│   └── gan-pokemon-study.pdf
├── results/
│   └── images/
├── src/
│   ├── eda.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── dcgan.py
│   ├── wgan_gp.py
│   ├── train.py
│   ├── evaluation.py
│   └── utils_gan.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ranuyay/pokemon-gan-sprite-generator.git
cd pokemon-gan-sprite-generator
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Core Dependencies

* TensorFlow
* NumPy
* pandas
* matplotlib
* Pillow
* SciPy
* d2l
* IPython

---

## Running the Project

Typical workflow:

1. Run exploratory data analysis in `src/eda.py`
2. Preprocess images using `src/preprocessing.py`
3. Build the TensorFlow dataset with `src/dataset.py`
4. Initialize models from `src/dcgan.py` or `src/wgan_gp.py`
5. Train models using `src/train.py` or `src/wgan_gp.py`
6. Evaluate results using utilities in `src/evaluation.py`

---

## Report

The full written report describing the experiments and findings is included here:

```
report/gan-pokemon-study.pdf
```

---

## Portfolio Context

This project demonstrates:

* deep learning experimentation
* GAN training instability analysis
* dataset preprocessing for generative models
* evaluation using distributional similarity metrics
* converting coursework into **clean, reproducible ML repositories**

---

## Future Improvements

Possible extensions include:

* experiment configuration management
* automatic generation of training sample grids
* CLI entry points for running experiments
* reproducible training scripts for each experiment
* clearer separation between **DCGAN** and **WGAN-GP** experiment pipelines

---

## Author

**Rania A. Hamid**

M.S. Data Analytics Candidate
University of Maryland Global Campus

Portfolio: https://ranuyay.github.io
GitHub: https://github.com/ranuyay
