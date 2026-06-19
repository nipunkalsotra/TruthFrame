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
from evaluation.main import generate_evaluation_report

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
# Parallel processing enabled for maximum throughput
MAX_CONCURRENT_REQUESTS = 5 
# Slight staggered start to avoid instant burst 429s
STAGGER_DELAY_SECONDS = 0.5


async def process_single_claim(index, row, loader, vision_engine, reasoning_engine, semaphore):
    """Orchestrates a single claim with rate limiting."""
    async with semaphore:
        logger.info(f"--- Processing Claim {index + 1} (User: {row['user_id']}) ---")

        # Step A: Vision Pre-screening (Synchronous OpenCV check)
        image_paths = str(row['image_paths']).split(';')
        first_image_id = image_paths[0]
        
        # Correctly resolve image path using the loader's map
        # Try full path then filename fallback
        image_path = loader.image_directory_map.get(first_image_id)
        if not image_path:
            image_path = loader.image_directory_map.get(Path(first_image_id).name)

        vision_results = {"valid": False, "risk_flags": ["image_not_found"]}
        if image_path:
            # Local CPU-bound work remains synchronous for stability
            vision_results = vision_engine.check_image_quality(str(image_path))

        # Step B: Async Reasoning Pipeline
        try:
            output: ClaimOutput = await reasoning_engine.run_pipeline_async(row.to_dict(), vision_results)
            logger.info(f"Result {index + 1}: {output.claim_status.value}")
            return output.model_dump()
        except Exception as e:
            logger.error(f"Failed to process claim {index + 1}: {e}")
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
        logger.info(f"Starting parallel processing for {total_claims} claims with live-sorting...")

        # 3. Setup Paths and FRESH START (Delete old files)
        project_root = loader.dataset_path.parent
        output_path = project_root / "output.csv"
        dataset_output_path = loader.dataset_path / "output.csv"
        report_path = Path(__file__).parent / "evaluation" / "evaluation_report.md"
        
        for p in [output_path, dataset_output_path, report_path]:
            if p.exists():
                p.unlink()
                logger.info(f"Cleared old file: {p.name}")
        
        # 4. Create Semaphore for Global Rate Limiting
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        # 5. Create Tasks
        tasks = []
        for index, row in loader.claims_df.iterrows():
            tasks.append(
                process_single_claim(index, row, loader, vision_engine, reasoning_engine, semaphore)
            )

        # 6. Execute, Live-Sort, and Save in Real-Time
        results_list = []
        for task in asyncio.as_completed(tasks):
            result = await task
            
            if result:
                results_list.append(result)
                
                # LIVE SORTING: Arrange results_list by user_id numerically
                # This ensures output.csv is ALWAYS in order even during parallel runs
                sorted_results = sorted(
                    results_list, 
                    key=lambda x: int(''.join(filter(str.isdigit, x['user_id']))) if any(c.isdigit() for c in x['user_id']) else x['user_id']
                )
                
                # Overwrite CSV with current sorted state
                res_df = pd.DataFrame(sorted_results)
                res_df.to_csv(output_path, index=False)
                res_df.to_csv(dataset_output_path, index=False)
                
                logger.info(f"Progress: {len(results_list)}/{total_claims} (Live-Sorted in {output_path})")
                
                # Update Evaluation Report in Real-Time
                try:
                    metrics = reasoning_engine.metrics
                    metrics["total_processing_time_sec"] = round(time.perf_counter() - start_time_global, 2)
                    generate_evaluation_report(loader.dataset_path, metrics, is_final=False)
                except Exception as e:
                    logger.warning(f"Could not update real-time report: {e}")

        if results_list:
            logger.info(f"SUCCESS: Total {len(results_list)} results finalized at {output_path}")
            
            # Final Sort: Ensure output.csv is in ascending order by user_id for submission
            logger.info("Sorting final output by user_id...")
            final_df = pd.DataFrame(results_list)
            # Handle numeric user_ids if necessary, otherwise string sort
            if 'user_id' in final_df.columns:
                # Try to extract number for natural sorting if format is user_001
                final_df['sort_key'] = final_df['user_id'].str.extract('(\d+)').astype(float)
                final_df = final_df.sort_values('sort_key').drop(columns=['sort_key'])
                
                final_df.to_csv(output_path, index=False)
                final_df.to_csv(dataset_output_path, index=False)
                logger.info("Final output sorted successfully.")
        else:
            logger.warning("No valid results were generated.")
    else:
        logger.warning("No claims found in dataset/claims.csv")

    # 6. Finalize Evaluation Report
    logger.info("--- Finalizing Evaluation Report ---")
    
    # Inject total time into metrics
    metrics = reasoning_engine.metrics
    metrics["total_processing_time_sec"] = round(time.perf_counter() - start_time_global, 2)
    
    generate_evaluation_report(loader.dataset_path, metrics, is_final=True)


if __name__ == "__main__":
    start_time_global = time.perf_counter()
    try:
        asyncio.run(main_orchestrator())
    except Exception as e:
        logger.error(f"Orchestrator crashed: {e}")
    end_time = time.perf_counter()
    logger.info(f"Total processing time: {end_time - start_time_global:.2f} seconds")