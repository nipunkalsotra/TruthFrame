import sys
from pathlib import Path
import logging
from data_loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("TruthFrame Phase 1: Initializing...")
    loader = DataLoader()
    loader.load_all_data()
    logger.info("Phase 1: Data Ingestion & Preprocessing - SUCCESS")
    return loader

if __name__ == "__main__":
    main()
