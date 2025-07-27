import cv2
import logging
import numpy as np
from io import BytesIO
from pathlib import Path
from datetime import datetime

__all__ = [
    "ensure_bytesio",
    "get_with_fallback",
    "debug_save_extracted_faces",
    "preprocess_image",
    "get_largest_face",
]

# ================================
# Logging Utilities
# ================================


def get_safe_logger(logger=None):
    """
    Return the given logger, or a fallback logger if None is provided.
    """
    if logger:
        return logger
    null_logger = logging.getLogger("null_logger")
    null_logger.addHandler(logging.NullHandler())
    return null_logger


# ================================
# General Utilities
# ================================


def ensure_bytesio(file):
    """
    Ensure that the given file is a BytesIO object.

    Args:
        file (BytesIO or file-like): File to convert.

    Returns:
        BytesIO: File as a BytesIO object.
    """
    if isinstance(file, BytesIO):
        file.seek(0)
        return file
    return BytesIO(file.read())


def get_with_fallback(mapping, key, default_key, name, logger=None):
    """
    Retrieve a value from a mapping with fallback support.

    If the requested key is missing, the fallback key is used (if available).
    This method does not treat None values as invalid — only missing keys trigger fallback.

    Args:
        mapping (dict): The dictionary to look into.
        key (str): The requested key.
        default_key (str): The fallback key to use if the requested key is missing.
        name (str): A label used in logs for clarity (e.g., "model key").

    Returns:
        any: The value from mapping[key], mapping[default_key], or None if both are missing.
    """
    logger = get_safe_logger(logger)

    if key in mapping:
        return mapping[key]

    if default_key in mapping:
        logger.warning(f"Invalid {name} '{key}', falling back to '{default_key}'.")
        return mapping[default_key]

    logger.error(f"Invalid {name} '{key}' and fallback '{default_key}' not found.")
    return "__INVALID__"


# ================================
# Face Matching Debug Toggle
# ================================


def debug_face_matcher():
    """
    Determine whether face matcher debugging is enabled.

    This function attempts to read the DEBUG_FACE_MATCHER flag from Django settings.
    If Django is not available (e.g., in a standalone script), it falls back to True.

    Returns:
        bool: True if debugging is enabled, otherwise False.
    """
    try:
        from django.conf import settings

        return getattr(settings, "DEBUG_FACE_MATCHER", False)
    except ImportError:
        # Fallback for non-Django environments
        return True


# ================================
# Image Processing Utilities
# ================================


def preprocess_image(img, label="unknown", normalize_brightness=False, logger=None):
    """
    Preprocess an image by normalizing brightness, ensuring data type, and converting to RGB if needed.

    Args:
        img (np.ndarray): Input image
        label (str): A label used in logs (e.g., "id_card", "selfie")
        normalize_brightness (bool): Whether to auto-adjust brightness if needed

    Returns:
        np.ndarray: The preprocessed image
    """
    logger = get_safe_logger(logger)

    # 1. Normalize brightness (if needed)
    if normalize_brightness:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        if brightness < 50 or brightness > 240:
            logger.warning(
                f"{label.capitalize()} image has extreme brightness! Avg={brightness:.2f}"
            )

        if brightness < 100:
            img = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
            logger.info(f"Brightened {label} image (avg={brightness:.2f})")
        elif brightness > 200:
            img = cv2.convertScaleAbs(img, alpha=0.9, beta=-30)
            logger.info(f"Darkened {label} image (avg={brightness:.2f})")
        else:
            logger.info(
                f"Skipped brightness normalization for {label} (avg={brightness:.2f})"
            )

    # 2. Normalize data range if float image
    if img.max() <= 1.0:
        img = img * 255

    # 3. Ensure uint8
    img = img.astype(np.uint8)

    # 4. Fix grayscale image (2D or single-channel)
    if len(img.shape) == 2:  # shape = (H, W)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:  # shape = (H, W, 1)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img


def get_largest_face(faces):
    """
    Get the largest face from a list of detected faces based on bounding box area.

    Args:
        faces (list): A list of face dicts with "facial_area" containing "w" and "h".

    Returns:
        dict or None: The largest face dict, or None if no faces provided.
    """
    return (
        max(faces, key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"])
        if faces
        else None
    )


# ================================
# Debug Helpers (Dev Only)
# ================================


def debug_save_extracted_faces(id_card_face_img, selfie_face_img, logger=None):
    """
    Save the extracted face images to the Desktop for debugging purposes.

    This function runs only if debugging is enabled — either via
    the `DEBUG_FACE_MATCHER` Django setting or when running in a standalone script
    (where it's enabled by default using the `debug_face_matcher()` utility).

    Args:
        id_card_face_img (np.ndarray): The extracted face from ID card image
        selfie_face_img (np.ndarray): The extracted face from selfie image
    """
    logger = get_safe_logger(logger)
    logger.info("Saving extracted faces to the Desktop for debugging.")

    # Get path to current user's Desktop (Linux/Mac/Windows compatible)
    desktop_path = Path.home() / "Desktop"
    base_folder = desktop_path / "extracted_faces"
    id_card_folder = base_folder / "from_id_card"
    selfie_folder = base_folder / "from_selfie"
    id_card_folder.mkdir(parents=True, exist_ok=True)
    selfie_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    id_card_path = id_card_folder / f"id_card_face_{timestamp}.jpg"
    selfie_path = selfie_folder / f"selfie_face_{timestamp}.jpg"

    cv2.imwrite(str(id_card_path), cv2.cvtColor(id_card_face_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(selfie_path), cv2.cvtColor(selfie_face_img, cv2.COLOR_RGB2BGR))
