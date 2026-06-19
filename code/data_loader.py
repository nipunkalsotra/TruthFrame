import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.code_dir = Path(__file__).parent
        self.dataset_path = self.code_dir.parent / "dataset"
        self.user_history_cache: Dict[str, Dict] = {}
        self.evidence_requirements_cache: Dict[str, Dict] = {}
        self.claims_df: Optional[pd.DataFrame] = None
        self.image_cache: Dict[str, Image.Image] = {}
        self.image_directory_map: Dict[str, Path] = {}
        logger.info(f"DataLoader initialized. Dataset at: {self.dataset_path}")
    
    def load_all_data(self) -> None:
        logger.info("Starting zero-redundancy data loading...")
        self._load_user_history()
        self._load_evidence_requirements()
        self._load_claims()
        self._index_image_directory()
        logger.info("Data loading complete.")
    
    def _load_user_history(self) -> None:
        path = self.dataset_path / "user_history.csv"
        if path.exists():
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                self.user_history_cache[str(row['user_id'])] = row.to_dict()
            logger.info(f"Cached {len(self.user_history_cache)} user history records.")
    
    def _load_evidence_requirements(self) -> None:
        path = self.dataset_path / "evidence_requirements.csv"
        if path.exists():
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                key = f"{row['claim_object']}_{row['applies_to']}"
                self.evidence_requirements_cache[key] = row.to_dict()
            logger.info(f"Cached {len(self.evidence_requirements_cache)} evidence requirements.")
    
    def _load_claims(self) -> None:
        path = self.dataset_path / "claims.csv"
        if path.exists():
            self.claims_df = pd.read_csv(path)
            logger.info(f"Loaded {len(self.claims_df)} claims.")
    
    def _index_image_directory(self) -> None:
        images_path = self.dataset_path / "images"
        if images_path.exists():
            for image_file in images_path.rglob("*"):
                if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.image_directory_map[image_file.name] = image_file
            logger.info(f"Indexed {len(self.image_directory_map)} images.")

    def get_user_history(self, user_id: str) -> Optional[Dict]:
        return self.user_history_cache.get(str(user_id))
    
    def get_evidence_requirement(self, object_type: str, issue_family: str) -> Optional[Dict]:
        res = self.evidence_requirements_cache.get(f"{object_type}_{issue_family}")
        return res if res else self.evidence_requirements_cache.get(f"all_{issue_family}")

    def get_image(self, image_id: str) -> Optional[Image.Image]:
        image_name = Path(image_id).name
        if image_name in self.image_cache:
            return self.image_cache[image_name]
        full_path = self.image_directory_map.get(image_name)
        if not full_path: return None
        try:
            image = Image.open(full_path)
            self.image_cache[image_name] = image
            return image
        except Exception:
            return None

    def get_claims_batch(self, batch_size: int = 32):
        if self.claims_df is not None:
            for i in range(0, len(self.claims_df), batch_size):
                yield self.claims_df.iloc[i:i + batch_size].to_dict('records')
