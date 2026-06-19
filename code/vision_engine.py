"""
Vision Engine: CV Pre-screening Module
Deterministic image quality checks to filter out "garbage" images before VLM calls.
This saves money and latency by catching blur, poor lighting, and missing objects locally.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisionEngine:
    """
    Performs deterministic Computer Vision checks on images.
    Returns a risk assessment before sending to expensive VLM.
    """
    
    # Thresholds for image quality checks
    BLUR_THRESHOLD = 100  # Laplacian variance threshold
    BRIGHTNESS_MIN = 30   # Minimum average pixel intensity
    BRIGHTNESS_MAX = 225  # Maximum average pixel intensity
    CONTRAST_THRESHOLD = 20  # Minimum standard deviation for contrast
    
    def __init__(self):
        logger.info("VisionEngine initialized.")
    
    def check_image_quality(self, image_path: str) -> Dict:
        """
        Perform comprehensive quality checks on an image.
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return {
                    "valid": False,
                    "risk_flags": ["image_not_found"],
                    "confidence_score": 0.0,
                }
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Perform checks
            blur_score = self._check_blur(gray)
            brightness_score = self._check_brightness(gray)
            contrast_score = self._check_contrast(gray)
            
            # Determine if image is valid and collect risk flags
            risk_flags = []
            is_valid = True
            
            if blur_score < self.BLUR_THRESHOLD:
                risk_flags.append("blurry_image")
                is_valid = False
            
            if brightness_score < self.BRIGHTNESS_MIN:
                risk_flags.append("low_light")
                is_valid = False
            elif brightness_score > self.BRIGHTNESS_MAX:
                risk_flags.append("high_glare")
                is_valid = False
            
            if contrast_score < self.CONTRAST_THRESHOLD:
                risk_flags.append("low_contrast")
                is_valid = False
            
            # Calculate confidence score (0-1)
            confidence = self._calculate_confidence(blur_score, brightness_score, contrast_score)
            
            return {
                "valid": is_valid,
                "risk_flags": risk_flags if risk_flags else ["none"],
                "blur_score": float(blur_score),
                "brightness_score": float(brightness_score),
                "contrast_score": float(contrast_score),
                "confidence_score": float(confidence),
            }
        
        except Exception as e:
            logger.error(f"Error checking image {image_path}: {str(e)}")
            return {
                "valid": False,
                "risk_flags": ["image_processing_error"],
                "confidence_score": 0.0,
            }
    
    def _check_blur(self, gray_image: np.ndarray) -> float:
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        return laplacian.var()
    
    def _check_brightness(self, gray_image: np.ndarray) -> float:
        return np.mean(gray_image)
    
    def _check_contrast(self, gray_image: np.ndarray) -> float:
        return np.std(gray_image)
    
    def _calculate_confidence(self, blur: float, brightness: float, contrast: float) -> float:
        # Normalize scores to 0-1 range for a simple confidence metric
        blur_norm = min(blur / self.BLUR_THRESHOLD, 1.0)
        brightness_norm = 1.0 - abs(brightness - 127.5) / 127.5
        contrast_norm = min(contrast / 50.0, 1.0)
        
        # Weighted average (blur is most important)
        return (blur_norm * 0.5 + brightness_norm * 0.3 + contrast_norm * 0.2)
