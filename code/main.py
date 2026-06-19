
import sys
import os
import pandas as pd
from pathlib import Path
import logging
from dotenv import load_dotenv

from data_loader import DataLoader
from vision_engine import VisionEngine
from reasoning_engine import ReasoningEngine
from models import ClaimOutput

# Load environment variables
load_dotenv()

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_claims():
    logger.info("TruthFrame: Initializing System...")
    
    # 1. Data Ingestion
    loader = DataLoader()
    loader.load_all_data()
    
    # 2. Initialize Engines
    vision_engine = VisionEngine()
    reasoning_engine = ReasoningEngine(loader)
    
    results = []
    
    if loader.claims_df is not None:
        logger.info(f"Processing {len(loader.claims_df)} claims...")
        
        for index, row in loader.claims_df.iterrows():
            logger.info(f"--- Processing Claim {index + 1} (User: {row['user_id']}) ---")
            
            # Step A: Vision Pre-screening
            # For simplicity, we check the first image path provided
            image_paths = row['image_paths'].split(';')
            first_image_id = image_paths[0]
            image_path = loader.image_directory_map.get(Path(first_image_id).name)
            
            vision_results = {"valid": False, "risk_flags": ["image_not_found"]}
            if image_path:
                vision_results = vision_engine.check_image_quality(str(image_path))
            
            # Step B: Reasoning Pipeline (Text + VLM + Decision)
            try:
                output: ClaimOutput = reasoning_engine.run_pipeline(row.to_dict(), vision_results)
                results.append(output.model_dump())
                logger.info(f"Result: {output.claim_status.value} - {output.claim_status_justification[:50]}")
            except Exception as e:
                logger.error(f"Failed to process claim {index}: {e}")
        
        # 3. Save Output
        if results:
            output_df = pd.DataFrame(results)
            output_path = loader.dataset_path / "output.csv"
            output_df.to_csv(output_path, index=False)
            logger.info(f"SUCCESS: Results saved to {output_path}")
    else:
        logger.warning("No claims found in dataset/claims.csv")

if __name__ == "__main__":
    process_claims()
