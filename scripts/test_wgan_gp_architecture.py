from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import tensorflow as tf
from src.wgan_gp import build_wgan_generator, build_wgan_critic


def main():
    latent_dim = 100
    batch_size = 4

    print("Building WGAN-GP generator...")
    generator = build_wgan_generator(latent_dim=latent_dim, base_filters=64)

    print("Building WGAN-GP critic...")
    critic = build_wgan_critic(base_filters=64)

    noise = tf.random.normal((batch_size, 1, 1, latent_dim))
    fake_images = generator(noise, training=False)
    critic_scores = critic(fake_images, training=False)

    print("\nGenerator summary:")
    generator.summary()

    print("\nCritic summary:")
    critic.summary()

    print("\nSanity checks:")
    print("Noise shape:         ", noise.shape)
    print("Generated shape:     ", fake_images.shape)
    print("Critic output shape: ", critic_scores.shape)
    print("Generated min/max:   ", float(tf.reduce_min(fake_images)), float(tf.reduce_max(fake_images)))

    assert fake_images.shape == (batch_size, 64, 64, 3), "Generator output shape is wrong"
    assert critic_scores.shape == (batch_size, 1, 1, 1), "Critic output shape is wrong"

    print("\nWGAN-GP architecture test passed.")


if __name__ == "__main__":
    main()