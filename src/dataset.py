import os
import tensorflow as tf

from src.preprocessing import preprocess_image

def tf_preprocess(filepath, label):
    img = tf.py_function(
        func=lambda f: preprocess_image(f.numpy().decode('utf-8')),
        inp=[filepath],
        Tout=tf.float32
    )  
    img.set_shape((64, 64, 3))  # Set static shape for TensorFlow
    return img, label

def get_valid_classes(data_dir, min_images=30):
    valid_classes = sorted([
        class_name
        for class_name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, class_name))
        and len(os.listdir(os.path.join(data_dir, class_name))) >= min_images
    ])
    return valid_classes


def get_filepaths_and_labels(data_dir, min_images=30):
    valid_classes = get_valid_classes(data_dir, min_images=min_images)

    filepaths = []
    labels = []

    for label, class_name in enumerate(valid_classes):
        class_path = os.path.join(data_dir, class_name)
        for fname in sorted(os.listdir(class_path)):
            if fname.lower().endswith(".png"):
                filepaths.append(os.path.join(class_path, fname))
                labels.append(label)

    return filepaths, labels, valid_classes


def build_dataset(
    data_dir,
    batch_size=256,
    min_images=30,
    shuffle=True,
    seed=42
):
    filepaths, labels, valid_classes = get_filepaths_and_labels(
        data_dir,
        min_images=min_images
    )

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(filepaths), seed=seed)

    dataset = dataset.map(
        tf_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, filepaths, labels, valid_classes