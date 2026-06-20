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
    OBJECT_SALIENCY_THRESHOLD = 0.05  # Min ratio of "object-like" pixels
    
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
            object_score, has_object = self._check_object_presence(image)
            
            # Determine if image is valid and collect risk flags
            risk_flags = []
            is_valid = True
            
            if blur_score < self.BLUR_THRESHOLD:
                risk_flags.append("blurry_image")
                is_valid = False
            
            if brightness_score < self.BRIGHTNESS_MIN or brightness_score > self.BRIGHTNESS_MAX or contrast_score < self.CONTRAST_THRESHOLD:
                risk_flags.append("low_light_or_glare")
                is_valid = False
                
            if not has_object:
                risk_flags.append("wrong_object")
                is_valid = False
            
            # Calculate confidence score (0-1)
            confidence = self._calculate_confidence(blur_score, brightness_score, contrast_score, object_score)
            
            return {
                "valid": is_valid,
                "risk_flags": risk_flags if risk_flags else ["none"],
                "blur_score": float(blur_score),
                "brightness_score": float(brightness_score),
                "contrast_score": float(contrast_score),
                "object_saliency_score": float(object_score),
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
    
    def _check_object_presence(self, image: np.ndarray) -> Tuple[float, bool]:
        """
        Detects if a significant object is present using edge density and contour analysis.
        This acts as a lightweight CV pre-screen before expensive VLM calls.
        """
        # 1. Edge Detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 2. Morphological closing to join edges into "objects"
        kernel = np.ones((5,5), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 3. Calculate Edge Density
        edge_density = np.sum(closed_edges > 0) / (image.shape[0] * image.shape[1])
        
        # 4. Contour Analysis (find the largest "blob")
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area_ratio = 0
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            max_area = cv2.contourArea(max_contour)
            max_area_ratio = max_area / (image.shape[0] * image.shape[1])
            
        # Combine edge density and max blob size for an "objectness" score
        objectness_score = (edge_density * 0.4 + max_area_ratio * 0.6)
        has_object = objectness_score > self.OBJECT_SALIENCY_THRESHOLD
        
        return objectness_score, has_object

    def _calculate_confidence(self, blur: float, brightness: float, contrast: float, object_score: float) -> float:
        # Normalize scores to 0-1 range for a simple confidence metric
        blur_norm = min(blur / self.BLUR_THRESHOLD, 1.0)
        brightness_norm = 1.0 - abs(brightness - 127.5) / 127.5
        contrast_norm = min(contrast / 50.0, 1.0)
        object_norm = min(object_score / (self.OBJECT_SALIENCY_THRESHOLD * 2), 1.0)
        
        # Weighted average (blur and object presence are most important)
        return (blur_norm * 0.4 + object_norm * 0.3 + brightness_norm * 0.2 + contrast_norm * 0.1)
