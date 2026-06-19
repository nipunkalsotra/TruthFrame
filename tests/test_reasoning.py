"""
Test script for TruthFrame Phase 2: Reasoning Engine.
Validates the integration of Textual Analysis and Visual Analysis (VLM).
"""

import sys
import os
from pathlib import Path
import logging
from dotenv import load_dotenv

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

from data_loader import DataLoader
from vision_engine import VisionEngine
from reasoning_engine import ReasoningEngine
from models import ClaimOutput

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("TestReasoning")

def test_reasoning_pipeline():
    logger.info("🚀 Starting Phase 2: Reasoning Engine Integration Test...")
    
    # 1. Initialize Components
    loader = DataLoader()
    loader.load_all_data()
    
    vision = VisionEngine()
    reasoning = ReasoningEngine(loader)
    
    # 2. Select a few sample claims to test
    if loader.claims_df is not None and not loader.claims_df.empty:
        # Test the first 3 claims
        test_samples = loader.claims_df.head(3)
        
        for index, row in test_samples.iterrows():
            logger.info(f"\n--- Testing Claim {index + 1} (User: {row['user_id']}) ---")
            logger.info(f"User Claim: {row['user_claim']}")
            
            # Step A: Vision Pre-screening
            image_paths = row['image_paths'].split(';')
            first_image_id = image_paths[0]
            image_path = loader.image_directory_map.get(Path(first_image_id).name)
            
            vision_results = {"valid": False, "risk_flags": ["image_not_found"]}
            if image_path:
                vision_results = vision.check_image_quality(str(image_path))
                logger.info(f"Vision Pre-screen: Valid={vision_results['valid']}, Flags={vision_results['risk_flags']}")
            
            # Step B: Full Reasoning Pipeline
            try:
                output: ClaimOutput = reasoning.run_pipeline(row.to_dict(), vision_results)
                
                # Step C: Validation of Output
                logger.info("✅ Pipeline Execution: SUCCESS")
                logger.info(f"Extracted Issue: {output.issue_type.value}")
                logger.info(f"Extracted Part: {output.object_part.value}")
                logger.info(f"Claim Status: {output.claim_status.value}")
                logger.info(f"Severity: {output.severity.value}")
                logger.info(f"Justification: {output.claim_status_justification[:100]}...")
                
                # Basic assertions
                assert isinstance(output, ClaimOutput)
                assert output.user_id == str(row['user_id'])
                assert len(output.claim_status_justification) <= 500 
                
            except Exception as e:
                logger.error(f"❌ Pipeline Execution FAILED for claim {index}: {e}")
                raise e
                
        logger.info("\n✨ All reasoning engine tests passed successfully!")
    else:
        logger.error("❌ No claims found in dataset to test.")

if __name__ == "__main__":
    # Check for API Keys
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("❌ GOOGLE_API_KEY not found in environment. Please set it in .env file.")
        sys.exit(1)
        
    test_reasoning_pipeline()
