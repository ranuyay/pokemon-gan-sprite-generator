import os
import numpy as np
from PIL import Image

def get_dataset_summary(data_dir):
    total_images = sum(len(files) for _, _, files in os.walk(data_dir))
    total_classes = len(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )

    return {
        "total_images": total_images,
        "total_classes": total_classes
    }


def get_class_counts(data_dir):
    class_counts = []
    class_names = []

    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            class_counts.append(count)
            class_names.append(class_name)

    return class_names, class_counts


def get_all_filepaths(data_dir):
    all_filepaths = []

    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for fname in os.listdir(class_path):
                if fname.lower().endswith(".png"):
                    all_filepaths.append(os.path.join(class_path, fname))

    return all_filepaths


def sample_pixel_statistics(sample_paths, target_size=64):
    sample_pixels = []

    for path in sample_paths:
        img = Image.open(path).convert("RGB").resize((target_size, target_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        sample_pixels.append(arr)

    sample_pixels = np.array(sample_pixels)

    return {
        "pixel_min": sample_pixels.min(),
        "pixel_max": sample_pixels.max(),
        "mean_r": sample_pixels[:, :, :, 0].mean(),
        "mean_g": sample_pixels[:, :, :, 1].mean(),
        "mean_b": sample_pixels[:, :, :, 2].mean(),
        "std_r": sample_pixels[:, :, :, 0].std(),
        "std_g": sample_pixels[:, :, :, 1].std(),
        "std_b": sample_pixels[:, :, :, 2].std(),
    }


def get_background_statistics(sample_pixels):
    near_white = (sample_pixels > 0.9).all(axis=-1).mean()
    near_black = (sample_pixels < 0.1).all(axis=-1).mean()

    return {
        "near_white": near_white,
        "near_black": near_black
    }


def summarize_transparency(sample_paths):
    modes = {}
    transparency_count = 0
    no_transparency_count = 0

    for path in sample_paths:
        img = Image.open(path)
        mode = img.mode
        modes[mode] = modes.get(mode, 0) + 1

        if "transparency" in img.info:
            transparency_count += 1
        else:
            no_transparency_count += 1

    return {
        "modes": modes,
        "has_transparency": transparency_count,
        "no_transparency": no_transparency_count
    }


def summarize_image_sizes(sample_paths):
    widths, heights = [], []

    for path in sample_paths:
        img = Image.open(path)
        w, h = img.size
        widths.append(w)
        heights.append(h)

    widths = np.array(widths)
    heights = np.array(heights)

    return {
        "width_min": widths.min(),
        "width_max": widths.max(),
        "width_mean": widths.mean(),
        "width_median": np.median(widths),
        "height_min": heights.min(),
        "height_max": heights.max(),
        "height_mean": heights.mean(),
        "height_median": np.median(heights),
    }