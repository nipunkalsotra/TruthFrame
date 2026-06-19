import sys
import os
import pandas as pd
from pathlib import Path
import logging
import asyncio
import time
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

# Global Orchestrator Config
MAX_CONCURRENT_REQUESTS = 5  # Adjust based on your API quota

async def process_single_claim(index, row, loader, vision_engine, reasoning_engine, semaphore):
    """Orchestrates a single claim with rate limiting."""
    async with semaphore:
        logger.info(f"--- Processing Claim {index + 1} (User: {row['user_id']}) ---")
        
        # Step A: Vision Pre-screening (Synchronous OpenCV check)
        image_paths = str(row['image_paths']).split(';')
        first_image_id = image_paths[0]
        image_path = loader.image_directory_map.get(Path(first_image_id).name)
        
        vision_results = {"valid": False, "risk_flags": ["image_not_found"]}
        if image_path:
            # Local CPU-bound work remains synchronous for stability
            vision_results = vision_engine.check_image_quality(str(image_path))
        
        # Step B: Async Reasoning Pipeline
        try:
            output: ClaimOutput = await reasoning_engine.run_pipeline_async(row.to_dict(), vision_results)
            logger.info(f"Result {index+1}: {output.claim_status.value}")
            return output.model_dump()
        except Exception as e:
            logger.error(f"Failed to process claim {index+1}: {e}")
            return None

async def main_orchestrator():
    logger.info("TruthFrame: Initializing Async Orchestrator...")
    
    # 1. Data Ingestion
    loader = DataLoader()
    loader.load_all_data()
    
    # 2. Initialize Engines
    vision_engine = VisionEngine()
    reasoning_engine = ReasoningEngine(loader)
    
    if loader.claims_df is not None:
        total_claims = len(loader.claims_df)
        logger.info(f"Starting parallel processing for {total_claims} claims...")
        
        # 3. Create Semaphore for Global Rate Limiting
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        # 4. Create Tasks
        tasks = []
        for index, row in loader.claims_df.iterrows():
            tasks.append(process_single_claim(index, row, loader, vision_engine, reasoning_engine, semaphore))
        
        # 5. Execute and Gather Results
        results_list = await asyncio.gather(*tasks)
        
        # Filter out failed tasks
        valid_results = [r for r in results_list if r is not None]
        
        # 6. Save Output
        if valid_results:
            output_df = pd.DataFrame(valid_results)
            output_path = loader.dataset_path / "output.csv"
            output_df.to_csv(output_path, index=False)
            logger.info(f"SUCCESS: {len(valid_results)} results saved to {output_path}")
    else:
        logger.warning("No claims found in dataset/claims.csv")

if __name__ == "__main__":
    # Using perf_counter for reliable timing across all Python versions
    start_time = time.perf_counter()
    try:
        asyncio.run(main_orchestrator())
    except Exception as e:
        logger.error(f"Orchestrator crashed: {e}")
    end_time = time.perf_counter()
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
