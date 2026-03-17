from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.dataset import build_dataset
from src.wgan_gp import (
    build_wgan_generator,
    build_wgan_critic,
    train_wgan_gp,
)


def main():
    data_dir = PROJECT_ROOT / "data" / "pokemon"
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Building real dataset...")
    dataset, filepaths, labels, valid_classes = build_dataset(
        data_dir=str(data_dir),
        batch_size=64,
        min_images=30,
        shuffle=True,
        seed=42
    )

    print(f"Retained classes: {len(valid_classes)}")
    print(f"Retained images:  {len(filepaths)}")

    print("\nBuilding WGAN-GP models...")
    generator = build_wgan_generator(latent_dim=100, base_filters=64)
    critic = build_wgan_critic(base_filters=64)

    print("\nRunning 1-epoch WGAN-GP smoke training...")
    start = time.time()

    history, best_epoch, best_fid = train_wgan_gp(
        dataset=dataset,
        generator=generator,
        critic=critic,
        num_epochs=1,
        latent_dim=100,
        lr_g=0.0001,
        lr_c=0.0001,
        lambda_gp=10.0,
        n_critic=2,
        fid_interval=10,
        real_features_cache=None,
        evaluate_fid_fn=None,
        experiment_name="wgan_gp_smoke",
        model_dir=str(model_dir)
    )

    elapsed = time.time() - start

    print("\nTraining completed.")
    print("Elapsed seconds:", round(elapsed, 2))
    print("Best epoch:", best_epoch)
    print("Best FID:", best_fid)
    print("History keys:", list(history.keys()))
    print("Epochs logged:", len(history["critic_loss"]))


if __name__ == "__main__":
    main()