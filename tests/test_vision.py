"""
Test script for TruthFrame Phase 2: Vision Engine.
Validates deterministic CV pre-screening logic.
"""

import sys
from pathlib import Path
import logging

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

from vision_engine import VisionEngine
from data_loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestVision")

def test_vision_engine():
    logger.info("Starting Phase 2: Vision Engine Tests...")
    
    # 1. Initialize Engines
    vision = VisionEngine()
    loader = DataLoader()
    loader.load_all_data()
    
    # 2. Test with a sample image from the dataset
    if loader.image_directory_map:
        sample_image_name = list(loader.image_directory_map.keys())[0]
        sample_image_path = loader.image_directory_map[sample_image_name]
        
        logger.info(f"Testing quality check on: {sample_image_path}")
        result = vision.check_image_quality(str(sample_image_path))
        
        logger.info("Quality Check Result:")
        for key, value in result.items():
            logger.info(f"  {key}: {value}")
            
        if result["valid"]:
            logger.info("✅ Image passed quality check.")
        else:
            logger.info(f"⚠️ Image flagged with: {result['risk_flags']}")
            
        # 3. Validation
        assert "valid" in result
        assert "confidence_score" in result
        assert "risk_flags" in result
        logger.info("✅ Vision Engine test passed basic validation.")
    else:
        logger.error("❌ No images found in dataset to test.")

if __name__ == "__main__":
    test_vision_engine()
