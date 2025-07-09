import os
import cv2
import logging
import tempfile
import numpy as np
from PIL import Image
from io import BytesIO
from deepface import DeepFace

"""
Logs for TensorFlow and DeepFace:
"0" = all logs
"1" = filter INFO
"2" = filter INFO + WARNING
"3" = only show critical errors
"""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODELS = ["ArcFace", "Facenet512", "DeepFace", "Facenet", "VGG-Face", "Dlib", "DeepID", "OpenFace"]
THRESHOLD = [None, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

logger = logging.getLogger("requests")


class FaceMatcher:
    """
    FaceMatcher class to verify if two face images (selfie and cédula) belong to the same person using DeepFace.

    This class handles:
        - Loading file-like selfie and cédula images
        - Converting them into NumPy arrays
        - Extracting faces using DeepFace
        - Saving temporary images
        - Comparing both faces using a selected model

    Args:
        selfie_file (file-like): The selfie image file (e.g., from request.FILES)
        cedula_file (file-like): The cédula image file

    Methods:
        match_faces(model="ArcFace", threshold_override=None) -> bool
            Compares the extracted faces. Returns True if they match.

    Example:
        >>> matcher = FaceMatcher(cedula_file=cedula_file, selfie_file=selfie_file, user_email=user_email)
        >>> match = matcher.match_faces()
        >>> if match:
        >>>     print("Faces matched!")
        >>> else:
        >>>     print("Faces do NOT match.")

    Supported Models and their default thresholds:
        - ArcFace     (default threshold: 0.68)
        - Facenet512  (default threshold: 0.30)
        - DeepFace    (default threshold: 0.23)
        - Facenet     (default threshold: 0.40)
        - VGG-Face    (default threshold: 0.40)
        - Dlib        (default threshold: 0.60)
        - DeepID      (default threshold: 0.17)
        - OpenFace    (default threshold: 0.10)

    Notes:
        - Logs result details: user_email, distance, threshold, confidence percentage, and model.
        - You can enable saving extracted faces to the Desktop for debugging and
          manual verification by uncommenting the "DEBUG" block in "match_faces()".
    """

    def __init__(self, selfie_file, cedula_file, user_email=None):
        self.user_email = user_email
        try:
            # Convert to BytesIO (whether already BytesIO or not)
            self.selfie_stream = self._ensure_bytesio(selfie_file)
            self.cedula_stream = self._ensure_bytesio(cedula_file)

            # Convert to NumPy arrays
            self.selfie_np = np.array(Image.open(self.selfie_stream).convert("RGB"))
            self.cedula_np = np.array(Image.open(self.cedula_stream).convert("RGB"))
        except Exception as e:
            logger.error(f"[FaceMatcher] ({self.user_email}) Failed to load input files: {e}")
            raise

    def _ensure_bytesio(self, file):
        if isinstance(file, BytesIO):
            file.seek(0)
            return file
        return BytesIO(file.read())

    def match_faces(self, model=MODELS[0], threshold_override=THRESHOLD[0]):
        try:
            try:
                # Extract face from cedula
                cedula_faces = DeepFace.extract_faces(img_path=self.cedula_np, enforce_detection=True)
                if not cedula_faces:
                    logger.error(f"[FaceMatcher] ({self.user_email}) No face detected in cédula.")
                    return False

                # Extract face from selfie
                selfie_faces = DeepFace.extract_faces(img_path=self.selfie_np, enforce_detection=True)
                if not selfie_faces:
                    logger.error(f"[FaceMatcher] ({self.user_email}) No face detected in selfie.")
                    return False
            except Exception as e:
                logger.error(f"[FaceMatcher] ({self.user_email}) Error in face extraction: {e}")
                return False

            # Convert extracted faces to uint8 for OpenCV
            cedula_face = (cedula_faces[0]["face"] * 255).astype(np.uint8)
            selfie_face = (selfie_faces[0]["face"] * 255).astype(np.uint8)

            """======= DEBUG: Save extracted faces to the Desktop ======="""
            from pathlib import Path
            from datetime import datetime

            # Get path to current user's Desktop (Linux/Mac/Windows compatible)
            desktop_path = Path.home() / "Desktop"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            cedula_path = desktop_path / f"cedula_face_{timestamp}.jpg"
            selfie_path = desktop_path / f"selfie_face_{timestamp}.jpg"

            cv2.imwrite(str(cedula_path), cv2.cvtColor(cedula_face, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(selfie_path), cv2.cvtColor(selfie_face, cv2.COLOR_RGB2BGR))
            """=========================================================="""

            # Save both images to temp files
            with (
                tempfile.NamedTemporaryFile(suffix=".jpg") as cedula_temp,
                tempfile.NamedTemporaryFile(suffix=".jpg") as selfie_temp,
            ):

                cv2.imwrite(cedula_temp.name, cv2.cvtColor(cedula_face, cv2.COLOR_RGB2BGR))
                cv2.imwrite(selfie_temp.name, cv2.cvtColor(selfie_face, cv2.COLOR_RGB2BGR))

                try:
                    # Perform verification
                    result = DeepFace.verify(
                        img1_path=cedula_temp.name,
                        img2_path=selfie_temp.name,
                        model_name=model,
                        enforce_detection=True,
                    )
                except Exception as e:
                    logger.error(f"[FaceMatcher] ({self.user_email}) Error in face matching: {e}")
                    return False

            verified = result.get("verified", False)
            distance = result.get("distance", None)
            threshold = result.get("threshold", None)

            # Optional override for custom threshold
            if threshold_override is not None and distance is not None:
                verified = distance < threshold_override
                threshold = threshold_override

            confidence = round((1 - distance) * 100, 2) if distance is not None else None

            logger.info(
                f"[FaceMatcher] User: {self.user_email} | Result: {'MATCH' if verified else 'NO MATCH'} "
                f"| Confidence: {confidence}% | Distance: {distance:.4f} | Threshold: {threshold:.4f} | Model: {model}"
            )

            return verified

        except Exception as e:
            logger.error(f"[FaceMatcher] ({self.user_email}) An error occurred: {e}")
            return False
