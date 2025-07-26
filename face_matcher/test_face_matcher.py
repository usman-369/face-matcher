import os
import logging

from .core import FaceMatcher
from .logger import FaceMatcherLoggerAdapter

# --- Setup logging ---
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


# --- Test Function ---
def test_face_matcher():
    log.info("Running FaceMatcher test...")

    # Get image paths from one directory back
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    id_card_path = os.path.join(base_dir, "test_id_card.jpg")
    selfie_path = os.path.join(base_dir, "test_selfie.jpg")

    # Ensure test files exist
    if not os.path.isfile(id_card_path) or not os.path.isfile(selfie_path):
        log.error(
            "Test images not found. Make sure 'test_id_card.jpg' and 'test_selfie.jpg' exist one directory up."
        )
        return

    # Read the image files as bytes
    with open(id_card_path, "rb") as f1, open(selfie_path, "rb") as f2:
        matcher = FaceMatcher(f1, f2, user_email="test@example.com")
        result = matcher.match_faces()

        if result:
            log.info("Faces matched.")
        else:
            log.info("Faces did NOT match.")


if __name__ == "__main__":
    test_face_matcher()
