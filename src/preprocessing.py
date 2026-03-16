import numpy as np
from PIL import Image

def get_bounding_box(img, threshold=240):
    """
    Detect bounding box of subject pixels.
    For images with transparency: uses alpha channel.
    For images without transparency: uses white color threshold.
    """
    if 'transparency' in img.info:
        # Use alpha channel to find subject
        rgba = img.convert('RGBA')
        alpha = np.array(rgba)[:,:,3]
        is_subject = alpha > 0
    else:
        # Use color threshold to find non-white pixels
        rgb = np.array(img.convert('RGB'))
        is_subject = ~((rgb[:,:,0] > threshold) &
                       (rgb[:,:,1] > threshold) &
                       (rgb[:,:,2] > threshold))

    rows = np.any(is_subject, axis=1)
    cols = np.any(is_subject, axis=0)

    if not rows.any():
        # Edge case: no subject pixels found, return full image bounds
        h, w = is_subject.shape
        return 0, 0, w, h

    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    return col_min, row_min, col_max + 1, row_max + 1

def preprocess_direct(filepath, target_size=64):
    """Direct resize approach — composite over white, resize to target."""
    img = Image.open(filepath)
    rgba = img.convert('RGBA')
    background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
    composited = Image.alpha_composite(background, rgba)
    rgb = composited.convert('RGB')
    resized = rgb.resize((target_size, target_size), Image.LANCZOS)
    arr = np.array(resized, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    return arr


def preprocess_crop_pad(filepath, target_size=64, threshold=240):
    """Crop-and-pad approach — detect subject, crop, pad to square, resize."""
    img = Image.open(filepath)
    rgba = img.convert('RGBA')
    background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
    composited = Image.alpha_composite(background, rgba)

    # Get bounding box using original img for transparency info
    col_min, row_min, col_max, row_max = get_bounding_box(img, threshold)

    # Crop to subject bounding box
    cropped = composited.convert('RGB').crop((col_min, row_min, col_max, row_max))

    # Pad to square
    w, h = cropped.size
    max_dim = max(w, h)
    pad_left   = (max_dim - w) // 2
    pad_top    = (max_dim - h) // 2
    padded = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
    padded.paste(cropped, (pad_left, pad_top))

    # Resize to target
    resized = padded.resize((target_size, target_size), Image.LANCZOS)
    arr = np.array(resized, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    return arr

def preprocess_crop_pad_margin(filepath, target_size=64, threshold=240, margin=0.05):
    """Crop-and-pad with configurable margin around bounding box."""
    img = Image.open(filepath)
    rgba = img.convert('RGBA')
    background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
    composited = Image.alpha_composite(background, rgba)

    col_min, row_min, col_max, row_max = get_bounding_box(img, threshold)

    # Add margin based on bounding box dimensions
    w = col_max - col_min
    h = row_max - row_min
    margin_px = int(max(w, h) * margin)

    # Expand bounding box by margin, clamped to image boundaries
    img_w, img_h = img.size
    col_min = max(0, col_min - margin_px)
    row_min = max(0, row_min - margin_px)
    col_max = min(img_w, col_max + margin_px)
    row_max = min(img_h, row_max + margin_px)

    # Crop, pad, resize
    cropped = composited.convert('RGB').crop((col_min, row_min, col_max, row_max))
    w, h = cropped.size
    max_dim = max(w, h)
    pad_left = (max_dim - w) // 2
    pad_top  = (max_dim - h) // 2
    padded = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
    padded.paste(cropped, (pad_left, pad_top))
    resized = padded.resize((target_size, target_size), Image.LANCZOS)
    arr = np.array(resized, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    return arr

def preprocess_image(filepath, target_size=64, threshold=240, margin=0.05):
    """Main preprocessing pipeline: crop subject, pad to square, resize, normalize."""
    img = Image.open(filepath)

    # Convert to RGBA and composite over white background
    rgba = img.convert('RGBA')
    background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
    composited = Image.alpha_composite(background, rgba)

    # Detect subject bounding box using original img for transparency info
    col_min, row_min, col_max, row_max = get_bounding_box(img, threshold)

    # Add margin
    w = col_max - col_min
    h = row_max - row_min
    margin_px = int(max(w, h) * margin)
    img_w, img_h = img.size
    col_min = max(0, col_min - margin_px)
    row_min = max(0, row_min - margin_px)
    col_max = min(img_w, col_max + margin_px)
    row_max = min(img_h, row_max + margin_px)

    # Crop to bounding box
    cropped = composited.convert('RGB').crop((col_min, row_min, col_max, row_max))

    # Pad to square
    w, h = cropped.size
    max_dim = max(w, h)
    pad_left = (max_dim - w) // 2
    pad_top  = (max_dim - h) // 2
    padded = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
    padded.paste(cropped, (pad_left, pad_top))

    # Resize and normalize
    resized = padded.resize((target_size, target_size), Image.LANCZOS)
    arr = np.array(resized, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    return arr
