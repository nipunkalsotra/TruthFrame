import json
from pydantic import ValidationError
from typing import Dict, Any, List
import logging
from models import ClaimOutput, IssueTypeEnum, ObjectPartEnum, ClaimStatusEnum, SeverityEnum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchemaValidator:
    def __init__(self, groq_client, groq_model):
        self.groq_client = groq_client
        self.groq_model = groq_model
        
        # Deterministic mapping for common hallucinations (Gap 2.6)
        self.fuzzy_map = {
            "issue_type": {
                "cracked": "crack",
                "shattered": "glass_shatter",
                "dented": "dent",
                "scratched": "scratch",
                "broken": "broken_part",
                "missing": "missing_part",
                "wet": "water_damage",
                "flooded": "water_damage",
                "torn": "torn_packaging",
                "crushed": "crushed_packaging"
            },
            "object_part": {
                "glass": "windshield",
                "tire": "body", # Fallback to body if part not in enum
                "mirror": "side_mirror",
                "bumper": "front_bumper", # Default to front
                "keyboard_area": "keyboard",
                "monitor": "screen",
                "packaging": "box"
            }
        }

        # Allowed parts per object type to enforce strict cross-object integrity
        self.allowed_parts = {
            "car": {
                "front_bumper", "rear_bumper", "door", "hood", "windshield", 
                "side_mirror", "headlight", "taillight", "fender", "quarter_panel", 
                "body", "unknown"
            },
            "laptop": {
                "screen", "keyboard", "trackpad", "hinge", "lid", 
                "corner", "port", "base", "body", "unknown"
            },
            "package": {
                "box", "package_corner", "package_side", "seal", "label", 
                "contents", "item", "unknown"
            }
        }

    async def _call_groq_correction(self, prompt: str) -> Dict:
        try:
            response = await self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Groq correction call failed: {e}")
            return {}

    async def validate_and_correct(self, data: Dict[str, Any], original_row: Dict) -> ClaimOutput:
        claim_object = str(original_row.get("claim_object", "")).lower().strip()

        # Step 1: Pre-validation Deterministic Mapping (Gap 2.6)
        for field, mapping in self.fuzzy_map.items():
            current_val = str(data.get(field, "")).lower()
            if current_val in mapping:
                logger.info(f"Applying deterministic mapping for {field}: {current_val} -> {mapping[current_val]}")
                data[field] = mapping[current_val]

        # Step 1.5: Cross-Object Part Mappings & Validation
        part = str(data.get("object_part", "")).lower().strip()
        if claim_object == "package":
            if part == "corner":
                data["object_part"] = "package_corner"
                part = "package_corner"
            elif part in ("side", "flap"):
                data["object_part"] = "package_side"
                part = "package_side"
        elif claim_object == "car":
            if part == "glass":
                data["object_part"] = "windshield"
                part = "windshield"
            elif part == "mirror":
                data["object_part"] = "side_mirror"
                part = "side_mirror"

        # Verify against allowed parts per object type
        if claim_object in self.allowed_parts:
            if part not in self.allowed_parts[claim_object]:
                logger.warning(f"Part '{part}' is invalid for claim_object '{claim_object}'. Resetting to 'unknown'.")
                data["object_part"] = "unknown"

        try:
            # Attempt to validate with Pydantic
            validated_output = ClaimOutput(**data)
            return validated_output
        except ValidationError as e:
            logger.warning(f"Schema validation failed for claim {original_row['user_id']}: {e}")
            
            # Step 2: Extract problematic fields and their allowed values
            correction_needed = {}
            for error in e.errors():
                field = error["loc"][0]
                current_value = data.get(field)
                allowed_values = []
                if field == "issue_type":
                    allowed_values = [e.value for e in IssueTypeEnum]
                elif field == "object_part":
                    allowed_values = [e.value for e in ObjectPartEnum]
                elif field == "claim_status":
                    allowed_values = [e.value for e in ClaimStatusEnum]
                elif field == "severity":
                    allowed_values = [e.value for e in SeverityEnum]
                
                if allowed_values:
                    correction_needed[field] = {
                        "current_value": current_value,
                        "allowed_values": allowed_values
                    }
            
            if correction_needed:
                correction_prompt = f"""
                The following fields in a claim output are invalid. Please correct them by choosing the closest matching value from the allowed_values list for each field. Respond only with a JSON object containing the corrected fields.
                Original data: {data}
                Correction needed: {correction_needed}
                """
                corrected_fields = await self._call_groq_correction(correction_prompt)
                
                # Apply corrections and re-validate
                corrected_data = {**data, **corrected_fields}
                try:
                    validated_output = ClaimOutput(**corrected_data)
                    logger.info(f"Successfully self-corrected claim {original_row["user_id"]}")
                    return validated_output
                except ValidationError as re_e:
                    logger.error(f"Self-correction failed for claim {original_row["user_id"]}: {re_e}")
                    # Fallback to original data with 'unknown' for problematic fields if correction fails
                    for error in re_e.errors():
                        field = error["loc"][0]
                        if field in corrected_data:
                            if field == "issue_type": corrected_data[field] = IssueTypeEnum.UNKNOWN.value
                            elif field == "object_part": corrected_data[field] = ObjectPartEnum.UNKNOWN.value
                            elif field == "claim_status": corrected_data[field] = ClaimStatusEnum.NOT_ENOUGH_INFORMATION.value
                            elif field == "severity": corrected_data[field] = SeverityEnum.UNKNOWN.value
                    return ClaimOutput(**corrected_data)
            else:
                logger.error(f"Validation failed but no specific correction rules found for claim {original_row["user_id"]}")
                # Fallback to original data with 'unknown' for problematic fields
                corrected_data = data.copy()
                for error in e.errors():
                    field = error["loc"][0]
                    if field in corrected_data:
                        if field == "issue_type": corrected_data[field] = IssueTypeEnum.UNKNOWN.value
                        elif field == "object_part": corrected_data[field] = ObjectPartEnum.UNKNOWN.value
                        elif field == "claim_status": corrected_data[field] = ClaimStatusEnum.NOT_ENOUGH_INFORMATION.value
                        elif field == "severity": corrected_data[field] = SeverityEnum.UNKNOWN.value
                return ClaimOutput(**corrected_data)
