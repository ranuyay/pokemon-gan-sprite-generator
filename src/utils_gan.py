"""
Shared utilities for GAN experiments:
- FID feature extraction and computation
- checkpoint sample generation
- experiment logging and comparison plots
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import display
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input


# Load InceptionV3 once at import time
inception = InceptionV3(
    include_top=False,
    pooling="avg",
    input_shape=(75, 75, 3)
)


def extract_features(images, batch_size=512):
    """Extract InceptionV3 features for a batch of images."""
    all_features = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch_resized = tf.image.resize(batch, (75, 75))
        batch_preprocessed = preprocess_input(batch_resized * 255.0)
        features = inception.predict(batch_preprocessed, verbose=0)
        all_features.append(features)

    return np.concatenate(all_features, axis=0)


def compute_fid(real_features, fake_features):
    """Compute Fréchet Inception Distance between real and generated features."""
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    cov_product = sigma_real @ sigma_fake
    sqrt_cov = sqrtm(cov_product)

    if np.iscomplexobj(sqrt_cov):
        sqrt_cov = sqrt_cov.real

    fid = (
        np.sum((mu_real - mu_fake) ** 2)
        + np.trace(sigma_real + sigma_fake - 2 * sqrt_cov)
    )
    return float(fid)


def cache_real_features(dataset, num_samples=500):
    """Cache Inception features for a fixed sample of real images."""
    print("Extracting real image features for FID computation...")

    real_images = []
    for x, _ in dataset:
        real_images.append(x.numpy() / 2 + 0.5)
        if sum(batch.shape[0] for batch in real_images) >= num_samples:
            break

    real_images = np.concatenate(real_images, axis=0)[:num_samples]
    features = extract_features(real_images)

    print(f"Real features cached: {features.shape}")
    return features


def evaluate_fid(generator, latent_dim, real_features_cache,
                 num_samples=500, experiment_name="experiment"):
    """Generate fake images and compute FID against cached real features."""
    noise = tf.random.normal((num_samples, 1, 1, latent_dim))
    fake_images = generator(noise, training=False) / 2 + 0.5
    fake_features = extract_features(fake_images.numpy())

    fid = compute_fid(real_features_cache, fake_features)
    print(f"FID ({experiment_name}): {fid:.2f}")
    return fid


def make_image_grid(images, ncols=7):
    """Convert a batch of images into a single display grid."""
    n_images = len(images)
    nrows = n_images // ncols

    grid = tf.concat(
        [
            tf.concat([images[i * ncols + j] for j in range(ncols)], axis=1)
            for i in range(nrows)
        ],
        axis=0
    )
    return grid


def generate_best(generator, latent_dim, best_epoch,
                  experiment_name="experiment", num_images=21,
                  model_dir="models", show=True):
    """
    Load the best saved generator weights, generate a grid of images,
    and optionally display the result.
    """
    model_dir = Path(model_dir)
    weight_path = model_dir / f"best_G_{experiment_name}.weights.h5"

    if weight_path.exists():
        generator.load_weights(weight_path)

    noise = tf.random.normal((num_images, 1, 1, latent_dim))
    fake_images = generator(noise, training=False) / 2 + 0.5
    grid = make_image_grid(fake_images, ncols=7)

    if show:
        plt.figure(figsize=(8, 4))
        plt.imshow(grid)
        plt.axis("off")
        plt.title(f"Best Generated Images — {experiment_name} (Epoch {best_epoch})")
        plt.tight_layout()
        plt.show()

    return fake_images


RESULT_COLUMNS = [
    "Experiment",
    "Loss Type",
    "n",
    "lambda_gp",
    "n_critic",
    "Augmentation",
    "Best Epoch",
    "FID",
    "Wall Time (s)",
    "Compute Units",
    "Notes",
]

results_df = pd.DataFrame(columns=RESULT_COLUMNS)


def log_experiment(experiment_name, loss_type, n, lambda_gp, n_critic,
                   augmentation, best_epoch, fid, wall_time,
                   compute_units=None, notes=""):
    """Append one experiment result to the global results table."""
    global results_df

    row = {
        "Experiment": experiment_name,
        "Loss Type": loss_type,
        "n": n,
        "lambda_gp": lambda_gp if lambda_gp is not None else "-",
        "n_critic": n_critic,
        "Augmentation": augmentation,
        "Best Epoch": best_epoch,
        "FID": round(fid, 2),
        "Wall Time (s)": round(wall_time, 1),
        "Compute Units": compute_units if compute_units is not None else "-",
        "Notes": notes,
    }

    results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

    print(
        f"Logged: {experiment_name} | "
        f"FID {fid:.2f} | "
        f"Wall time {wall_time:.1f}s"
    )


def show_results():
    """Display all logged experiments sorted by FID."""
    global results_df

    if results_df.empty:
        print("No experiments logged yet.")
        return

    df_sorted = results_df.copy()
    df_sorted["FID"] = df_sorted["FID"].astype(float)
    df_sorted = df_sorted.sort_values("FID", ascending=True)

    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS (Sorted by FID — lower is better)")
    print("=" * 80)
    display(df_sorted)

    best = df_sorted.iloc[0]
    print("\n" + "=" * 80)
    print(f"BEST: {best['Experiment']}")
    print(f"   FID:        {best['FID']:.2f}")
    print(f"   Best Epoch: {best['Best Epoch']}")
    print(f"   Wall Time:  {best['Wall Time (s)']}s")
    print("=" * 80)


def plot_fid_comparison():
    """Plot FID comparison across logged experiments."""
    global results_df

    if results_df.empty:
        print("No experiments logged yet.")
        return

    df_sorted = results_df.copy()
    df_sorted["FID"] = df_sorted["FID"].astype(float)
    df_sorted = df_sorted.sort_values("FID", ascending=True)

    plt.figure(figsize=(14, 5))
    bars = plt.bar(range(len(df_sorted)), df_sorted["FID"],
                   color="steelblue", alpha=0.8)

    for bar, val in zip(bars, df_sorted["FID"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.xticks(
        range(len(df_sorted)),
        df_sorted["Experiment"],
        rotation=45,
        ha="right",
        fontsize=8
    )
    plt.ylabel("FID (lower is better)")
    plt.title("FID Comparison Across Experiments")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()