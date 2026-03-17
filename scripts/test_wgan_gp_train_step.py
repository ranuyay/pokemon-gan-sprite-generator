from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import tensorflow as tf

from src.wgan_gp import (
    build_wgan_generator,
    build_wgan_critic,
    train_critic_step,
    train_generator_step,
    make_wgan_optimizers,
)


def main():
    latent_dim = 100
    batch_size = 4
    lambda_gp = 10.0

    print("Building models...")
    generator = build_wgan_generator(latent_dim=latent_dim, base_filters=64)
    critic = build_wgan_critic(base_filters=64)

    print("Creating optimizers...")
    optimizer_c, optimizer_g = make_wgan_optimizers(lr_g=0.0001, lr_c=0.0001)

    print("Creating dummy real batch...")
    real_images = tf.random.uniform(
        shape=(batch_size, 64, 64, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=tf.float32
    )

    print("Running one critic step...")
    loss_c, wass_loss, gp = train_critic_step(
        real_images,
        critic,
        generator,
        optimizer_c,
        latent_dim,
        lambda_gp
    )

    print("Running one generator step...")
    loss_g = train_generator_step(
        generator,
        critic,
        optimizer_g,
        batch_size,
        latent_dim
    )

    print("\nOutputs:")
    print("Critic loss:      ", float(loss_c))
    print("Wasserstein loss: ", float(wass_loss))
    print("Gradient penalty: ", float(gp))
    print("Generator loss:   ", float(loss_g))

    assert tf.math.is_finite(loss_c), "Critic loss is not finite"
    assert tf.math.is_finite(wass_loss), "Wasserstein loss is not finite"
    assert tf.math.is_finite(gp), "Gradient penalty is not finite"
    assert tf.math.is_finite(loss_g), "Generator loss is not finite"

    print("\nWGAN-GP train-step test passed.")


if __name__ == "__main__":
    main()