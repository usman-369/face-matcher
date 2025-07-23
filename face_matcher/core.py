import os

"""
Logs for TensorFlow and DeepFace:
    "0" = all logs
    "1" = filter INFO
    "2" = filter INFO + WARNING
    "3" = only show critical errors

Note: This should remain at the top.
"""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import time
import tempfile
import numpy as np
from deepface import DeepFace
from django.conf import settings

from .constants import DETECTORS, MODELS, THRESHOLD
from .logger import FaceMatcherLoggerAdapter, logger
from .utils import (
    ensure_bytesio,
    get_with_fallback,
    debug_save_extracted_faces,
    preprocess_image,
    get_largest_face,
)

START_MSG = "======= FaceMatcher started. ======="
END_MSG = "======= FaceMatcher ended. ======="
ERROR_END_MSG = "======= FaceMatcher ended with error. ======="


class FaceMatcher:
    """
    FaceMatcher class to verify if two face images (cédula and selfie) belong to the same person using DeepFace.

    Features:
        - Loads cédula and selfie files and converts them to NumPy arrays
        - Extracts faces using configurable DeepFace detectors
        - Saves extracted faces to disk for debugging (if enabled)
        - Compares both faces using a chosen DeepFace model

    Args:
        cedula_file (file-like): Cédula image file
        selfie_file (file-like): Selfie image file (e.g., from request.FILES)
        user_email (str, optional): For logging context

    Example:
        >>> from face_matcher import FaceMatcher
        >>> matcher = FaceMatcher(cedula_file, selfie_file, user_email)
        >>> match = matcher.match_faces()
        >>> print("Matched!" if match else "No match")

    Notes:
        - `match_faces()` allows custom detector, model, and threshold via keys
        - Supported DeepFace models: ArcFace, Facenet512, DeepFace, VGG-Face, Dlib, etc.
        - Use `settings.DEBUG_FACE_MATCHER = True` to save faces to Desktop for debugging
    """

    def __init__(self, selfie_file, cedula_file, user_email=None):
        self.user_email = user_email
        self._cedula_face_dict = None
        self._selfie_face_dict = None

        # Custom logger adapter for FaceMatcher
        self.logger = FaceMatcherLoggerAdapter(logger, {"user_email": self.user_email})

        try:
            # Convert to BytesIO (if not already)
            self.selfie_stream = ensure_bytesio(selfie_file)
            self.cedula_stream = ensure_bytesio(cedula_file)

            # Read image bytes into OpenCV-compatible format (BGR)
            self.cedula_np = cv2.imdecode(np.frombuffer(self.cedula_stream.read(), np.uint8), cv2.IMREAD_COLOR)
            self.selfie_np = cv2.imdecode(np.frombuffer(self.selfie_stream.read(), np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            self.logger.error(f"Failed to load input files: {e}")
            raise

    def extract_face(self, image_np, detector, label="unknown", normalize_brightness=True):
        """
        Extract the largest face from an image using a DeepFace detector.

        Args:
            image_np (np.ndarray): The image to process
            label (str): Label used for logging (e.g., 'selfie', 'cédula')
            detector (str or None): Detector name (e.g., 'opencv', 'mtcnn'), or None to fallback through all
            normalize_brightness (bool): Whether to auto-correct brightness before detection

        Returns:
            dict or None: Dictionary with face data, or None if no face found
        """

        # Reuse face if already extracted
        if label == "cédula" and self._cedula_face_dict is not None:
            self.logger.debug("Using cached face for cédula")
            return self._cedula_face_dict
        if label == "selfie" and self._selfie_face_dict is not None:
            self.logger.debug("Using cached face for selfie")
            return self._selfie_face_dict

        image_np = preprocess_image(
            image_np, label=label, normalize_brightness=normalize_brightness, logger=self.logger
        )
        backends = [detector] if detector else [d for d in DETECTORS.values() if d is not None]

        for backend in backends:
            try:
                faces = DeepFace.extract_faces(img_path=image_np, detector_backend=backend, enforce_detection=True)
                if faces:
                    face = get_largest_face(faces)
                    if face:
                        face["face"] = preprocess_image(face["face"], label=f"{label} (extracted)", logger=self.logger)

                        # Cache face here
                        if label == "cédula":
                            self._cedula_face_dict = face
                        elif label == "selfie":
                            self._selfie_face_dict = face

                        area = face["facial_area"]
                        area_size = area["h"] * area["w"]
                        self.logger.info(
                            f"Face detection successful in {label} with '{backend}' "
                            f"(H={area['h']} | W={area['w']} | Area={area_size})"
                        )
                        return face
            except Exception as e:
                self.logger.warning(f"Face detection failed in {label} with '{backend}': {e}")

        if detector is None:
            self.logger.error(f"Error in face extraction: No face detected in {label} after all fallbacks.")
        else:
            self.logger.error(f"Error in face extraction: No face detected in {label} using detector '{detector}'.")
        return None

    def match_faces(self, model_key="0", threshold_key="0", detector_key="0"):
        """
        Perform face matching between cédula and selfie using DeepFace.

        Args:
            model_key (str): Key to select model from MODELS dict
            threshold_key (str): Key to select threshold from THRESHOLD dict
            detector_key (str): Key to select detector from DETECTORS dict

        Returns:
            bool: True if the faces match, False otherwise
        """
        try:
            start_time = time.time()
            self.logger.info(START_MSG)

            # Fallback: if the key doesn't exist
            model = get_with_fallback(MODELS, model_key, default_key="0", name="model key", logger=self.logger)
            threshold_override = get_with_fallback(
                THRESHOLD, threshold_key, default_key="0", name="threshold key", logger=self.logger
            )
            detector = get_with_fallback(
                DETECTORS, detector_key, default_key="0", name="detector key", logger=self.logger
            )

            if not model or not detector:
                self.logger.info(ERROR_END_MSG)
                return False

            # Extract face from cédula and selfie
            cedula_face_dict = self.extract_face(self.cedula_np, detector=detector, label="cédula")
            selfie_face_dict = self.extract_face(self.selfie_np, detector=detector, label="selfie")

            if not cedula_face_dict or not selfie_face_dict:
                self.logger.info(ERROR_END_MSG)
                return False

            cedula_face_img = cedula_face_dict["face"]
            selfie_face_img = selfie_face_dict["face"]

            # DEBUG: Save extracted faces to the Desktop if setting is enabled
            if getattr(settings, "DEBUG_FACE_MATCHER", False):
                debug_save_extracted_faces(cedula_face_img, selfie_face_img, logger=self.logger)

            # Save both images to temp files
            with (
                tempfile.NamedTemporaryFile(suffix=".jpg") as cedula_temp,
                tempfile.NamedTemporaryFile(suffix=".jpg") as selfie_temp,
            ):

                cv2.imwrite(cedula_temp.name, cedula_face_img)
                cv2.imwrite(selfie_temp.name, selfie_face_img)

                try:
                    # Perform verification
                    result = DeepFace.verify(
                        img1_path=cedula_temp.name,
                        img2_path=selfie_temp.name,
                        model_name=model,
                        enforce_detection=True,
                    )
                except Exception as e:
                    self.logger.error(f"Error in face matching: {e}")
                    self.logger.info(ERROR_END_MSG)
                    return False

            verified = result.get("verified", False)
            distance = result.get("distance", None)
            threshold = result.get("threshold", None)

            # Override for custom threshold
            if threshold_override is not None and distance is not None:
                verified = distance < threshold_override
                threshold = threshold_override

            confidence = round((1 - distance) * 100, 2) if distance is not None else None

            self.logger.info(
                f"Result: {'MATCH' if verified else 'NO MATCH'} | Confidence: {confidence}% "
                f"| Distance: {distance:.4f} | Threshold: {threshold:.4f} | Model: {model}"
            )

            end_time = time.time()
            elapsed_time = end_time - start_time
            self.logger.info(f"Face matching took {elapsed_time:.2f} seconds.")

            self.logger.info(END_MSG)
            return verified

        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            self.logger.info(ERROR_END_MSG)
            return False
