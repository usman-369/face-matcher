import os
import sys
import logging

from .core import FaceMatcher
from .logger import FaceMatcherLoggerAdapter

# Setup logging
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_FILE = os.path.join(LOG_DIR, "face_matcher.log")

# Ensure logger only gets configured once
if not logging.getLogger("face_matcher").hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler()],
    )

logger = logging.getLogger("face_matcher")
log = FaceMatcherLoggerAdapter(logger, extra={"user_email": "test@example.com"})


# Test Function
def test_face_matcher(id_card_path=None, selfie_path=None):
    log.info("Running FaceMatcher test...")

    # If no args provided, use defaults
    if not id_card_path or not selfie_path:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        id_card_path = os.path.join(base_dir, "test_id_card.jpg")
        selfie_path = os.path.join(base_dir, "test_selfie.jpg")
        log.info("No image paths provided. Using default test images.")

    # Validate image paths
    if not os.path.isfile(id_card_path) or not os.path.isfile(selfie_path):
        log.error(
            f"Image(s) not found.\n"
            f"ID card path: {id_card_path}\n"
            f"Selfie path: {selfie_path}\n"
            f"Make sure the files exist."
        )
        sys.exit(1)

    # Read the image files as bytes
    with open(id_card_path, "rb") as f1, open(selfie_path, "rb") as f2:
        matcher = FaceMatcher(f1, f2, user_email="test@example.com")
        result = matcher.match_faces()

        if result:
            log.info("Faces matched.")
        else:
            log.info("Faces did NOT match.")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        test_face_matcher(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 1:
        test_face_matcher()
    else:
        print("Usage:")
        print("  python -m face_matcher.test_face_matcher <id_card_path> <selfie_path>")
        print("  OR just run without arguments to use default test images.")
