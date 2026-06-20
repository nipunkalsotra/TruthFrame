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

        # Step A: Vision Pre-screening (Synchronous OpenCV check across all images)
        image_paths = str(row['image_paths']).split(';')
        
        valid_any = False
        all_risk_flags = set()
        best_confidence = 0.0
        details_list = []
        
        for img_id in image_paths:
            # Resolve image path using loader's map
            img_path = loader.image_directory_map.get(img_id)
            if not img_path:
                img_path = loader.image_directory_map.get(Path(img_id).name)
                
            if img_path:
                quality_res = vision_engine.check_image_quality(str(img_path))
                if quality_res.get("valid", False):
                    valid_any = True
                if "risk_flags" in quality_res:
                    for flag in quality_res["risk_flags"]:
                        if flag != "none":
                            all_risk_flags.add(flag)
                best_confidence = max(best_confidence, quality_res.get("confidence_score", 0.0))
                details_list.append(quality_res)
            else:
                all_risk_flags.add("image_not_found")
                
        # If no images were successfully loaded, mark as invalid
        if not details_list:
            vision_results = {
                "valid": False,
                "risk_flags": ["image_not_found"],
                "confidence_score": 0.0
            }
        else:
            vision_results = {
                "valid": valid_any,
                "risk_flags": list(all_risk_flags) if all_risk_flags else ["none"],
                "confidence_score": best_confidence
            }

        # Step B: Async Reasoning Pipeline
        try:
            output: ClaimOutput = await reasoning_engine.run_pipeline_async(row.to_dict(), vision_results)
            logger.info(f"Result {index + 1}: {output.claim_status.value}")
            return output.model_dump()
        except Exception as e:
            logger.error(f"Failed to process claim {index + 1}: {e}")
            return None


async def main_orchestrator(claims_filename: str = "claims.csv", is_eval_mode: bool = False):
    logger.info("TruthFrame: Initializing Async Orchestrator...")
    start_time_global = time.perf_counter()

    # 1. Data Ingestion
    loader = DataLoader()
    loader.load_all_data(claims_filename)

    # 2. Initialize Engines
    vision_engine = VisionEngine()
    reasoning_engine = ReasoningEngine(loader)

    if loader.claims_df is not None:
        total_claims = len(loader.claims_df)
        logger.info(f"Starting parallel processing for {total_claims} claims from {claims_filename} with live-sorting...")

        # 3. Setup Paths and FRESH START (Delete old files)
        project_root = loader.dataset_path.parent
        output_filename = "output_sample.csv" if is_eval_mode else "output.csv"
        output_path = project_root / output_filename
        dataset_output_path = loader.dataset_path / output_filename
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
                
                # Ensure correct column ordering
                columns_order = [
                    "user_id", "image_paths", "user_claim", "claim_object", 
                    "evidence_standard_met", "evidence_standard_met_reason", 
                    "risk_flags", "issue_type", "object_part", "claim_status", 
                    "claim_status_justification", "supporting_image_ids", 
                    "valid_image", "severity"
                ]
                # Filter to only existing columns in case of missing data in early iterations
                actual_columns = [col for col in columns_order if col in res_df.columns]
                res_df = res_df[actual_columns]
                
                res_df.to_csv(output_path, index=False)
                res_df.to_csv(dataset_output_path, index=False)
                
                logger.info(f"Progress: {len(results_list)}/{total_claims} (Live-Sorted in {output_filename})")
                
                # Update Evaluation Report in Real-Time
                try:
                    metrics = reasoning_engine.metrics
                    metrics["total_processing_time_sec"] = round(time.perf_counter() - start_time_global, 2)
                    generate_evaluation_report(loader.dataset_path, metrics, output_filename=output_filename, is_final=False)
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
                
                # Enforce exact column order
                columns_order = [
                    "user_id", "image_paths", "user_claim", "claim_object", 
                    "evidence_standard_met", "evidence_standard_met_reason", 
                    "risk_flags", "issue_type", "object_part", "claim_status", 
                    "claim_status_justification", "supporting_image_ids", 
                    "valid_image", "severity"
                ]
                final_df = final_df[columns_order]
                
                final_df.to_csv(output_path, index=False)
                final_df.to_csv(dataset_output_path, index=False)
                logger.info("Final output sorted successfully.")
        else:
            logger.warning("No valid results were generated.")
    else:
        logger.warning(f"No claims found in {claims_filename}")

    # 6. Finalize Evaluation Report
    logger.info("--- Finalizing Evaluation Report ---")
    
    # Inject total time into metrics
    metrics = reasoning_engine.metrics
    metrics["total_processing_time_sec"] = round(time.perf_counter() - start_time_global, 2)
    
    output_filename = "output_sample.csv" if is_eval_mode else "output.csv"
    generate_evaluation_report(loader.dataset_path, metrics, output_filename=output_filename, is_final=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TruthFrame Async Orchestrator")
    parser.add_argument("--file", type=str, default=None, help="Specific claims CSV file to process")
    parser.add_argument("--eval", action="store_true", help="Run in evaluation mode using sample_claims.csv")
    args = parser.parse_args()

    # Determine input filename
    if args.file:
        claims_filename = args.file
    elif args.eval:
        claims_filename = "sample_claims.csv"
    else:
        claims_filename = "claims.csv"

    # Determine if we are in evaluation mode
    is_eval = args.eval or (args.file and "sample" in args.file)

    try:
        asyncio.run(main_orchestrator(claims_filename, is_eval_mode=is_eval))
    except Exception as e:
        logger.error(f"Orchestrator crashed: {e}")