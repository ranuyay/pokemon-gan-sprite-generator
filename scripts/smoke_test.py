from pathlib import Path
import sys

# Ensure project root is on path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import tensorflow as tf
from PIL import Image
import numpy as np

from src.preprocessing import preprocess_image
from src.dcgan import build_generator, build_discriminator
from src.wgan_gp import build_wgan_generator, build_wgan_critic
from src.train import make_dcgan_loss, make_dcgan_optimizers
from src.dataset import tf_preprocess


def test_model_builds():
    print("Testing DCGAN model builds...")
    g = build_generator()
    d = build_discriminator()

    noise = tf.random.normal((2, 1, 1, 100))
    fake = g(noise, training=False)
    score = d(fake, training=False)

    print("  Generator output shape:", fake.shape)
    print("  Discriminator output shape:", score.shape)

    assert fake.shape == (2, 64, 64, 3)
    assert len(score.shape) == 4

    print("Testing WGAN-GP model builds...")
    wg = build_wgan_generator()
    wc = build_wgan_critic()

    fake_w = wg(noise, training=False)
    score_w = wc(fake_w, training=False)

    print("  WGAN generator output shape:", fake_w.shape)
    print("  WGAN critic output shape:", score_w.shape)

    assert fake_w.shape == (2, 64, 64, 3)
    assert len(score_w.shape) == 4


def test_preprocessing():
    print("Testing preprocessing pipeline...")

    temp_dir = PROJECT_ROOT / "temp_test"
    temp_dir.mkdir(exist_ok=True)

    img_path = temp_dir / "dummy.png"

    # Create a simple white image with a colored rectangle
    img = Image.new("RGB", (80, 50), (255, 255, 255))
    arr = np.array(img)
    arr[10:40, 20:60] = [255, 0, 0]
    Image.fromarray(arr).save(img_path)

    processed = preprocess_image(str(img_path))

    print("  Preprocessed image shape:", processed.shape)
    print("  Value range:", processed.min(), "to", processed.max())

    assert processed.shape == (64, 64, 3)
    assert processed.min() >= -1.0 - 1e-5
    assert processed.max() <= 1.0 + 1e-5

    # cleanup
    img_path.unlink(missing_ok=True)
    temp_dir.rmdir()


def test_training_helpers():
    print("Testing training helpers...")
    loss_fn = make_dcgan_loss()
    opt_d, opt_g = make_dcgan_optimizers()

    assert loss_fn is not None
    assert opt_d is not None
    assert opt_g is not None

    print("  Loss and optimizers created successfully.")


def test_tf_preprocess_wrapper():
    print("Testing tf_preprocess wrapper...")

    temp_dir = PROJECT_ROOT / "temp_test"
    temp_dir.mkdir(exist_ok=True)

    img_path = temp_dir / "dummy_tf.png"

    img = Image.new("RGB", (64, 64), (255, 255, 255))
    arr = np.array(img)
    arr[16:48, 16:48] = [0, 0, 255]
    Image.fromarray(arr).save(img_path)

    filepath = tf.constant(str(img_path))
    label = tf.constant(1)

    image_tensor, label_tensor = tf_preprocess(filepath, label)

    print("  tf_preprocess output shape:", image_tensor.shape)
    print("  label:", label_tensor.numpy())

    assert image_tensor.shape == (64, 64, 3)
    assert int(label_tensor.numpy()) == 1

    img_path.unlink(missing_ok=True)
    temp_dir.rmdir()


if __name__ == "__main__":
    print("Running smoke tests...\n")
    test_model_builds()
    test_preprocessing()
    test_training_helpers()
    test_tf_preprocess_wrapper()
    print("\nAll smoke tests passed.")