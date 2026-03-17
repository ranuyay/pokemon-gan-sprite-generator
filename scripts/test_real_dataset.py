from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.dataset import build_dataset


def main():
    data_dir = PROJECT_ROOT / "data" / "pokemon"

    print("Building dataset from:")
    print(data_dir)

    dataset, filepaths, labels, valid_classes = build_dataset(
        data_dir=str(data_dir),
        batch_size=32,
        min_images=30,
        shuffle=True,
        seed=42
    )

    print("\nDataset summary:")
    print("Number of retained classes:", len(valid_classes))
    print("Number of retained images: ", len(filepaths))
    print("Number of labels:          ", len(labels))

    for images, batch_labels in dataset.take(1):
        print("\nFirst batch:")
        print("Image batch shape:", images.shape)
        print("Label batch shape:", batch_labels.shape)
        print("Pixel range:      ", float(images.numpy().min()), "to", float(images.numpy().max()))
        print("Sample labels:    ", batch_labels.numpy()[:10])

    print("\nReal dataset pipeline test passed.")


if __name__ == "__main__":
    main()