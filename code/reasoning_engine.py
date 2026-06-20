import os
import io
import json
import asyncio
import random
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai import types
from groq import AsyncGroq
from PIL import Image

from models import (
    ClaimObjectEnum, IssueTypeEnum, ObjectPartEnum,
    ClaimStatusEnum, SeverityEnum, ClaimOutput
)
from schema_validator import SchemaValidator
from rate_limiter import GlobalRateLimiter

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global rate limiter to replace simple locks with dynamic token buckets
rate_limiter = GlobalRateLimiter()


class ReasoningEngine:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set. Create a .env file with GOOGLE_API_KEY=AIza... "
                "or run: export GOOGLE_API_KEY='AIza...'"
            )
        self.client = genai.Client(api_key=self.api_key)
        
        # Configure rate limits from env or defaults
        rate_limiter.set_limit(
            "gemini", 
            int(os.getenv("GEMINI_TPM_LIMIT", 1000000)), 
            int(os.getenv("GEMINI_RPM_LIMIT", 15))
        )
        rate_limiter.set_limit(
            "groq", 
            int(os.getenv("GROQ_TPM_LIMIT", 30000)), 
            int(os.getenv("GROQ_RPM_LIMIT", 30))
        )

        # Tiered Models (Operationally Superior Strategy)
        # Using 2.5-flash for maximum stability on Free Tier
        self.primary_model = "gemini-2.5-flash"
        self.critic_model = "gemini-2.5-flash"
        self.judge_model = "gemini-2.5-flash"
        self.fallback_model = "gemini-2.5-flash"

        # Groq Fallback (Primary for Logic to save Gemini Quota)
        self.groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        self.schema_validator = SchemaValidator(self.groq_client, self.groq_model)

        # Metrics tracking
        self.metrics = {
            "gemini_calls": 0,
            "groq_calls": 0,
            "gemini_tokens": 0,
            "groq_tokens": 0,
            "images_processed": 0,
            "self_corrections": 0
        }

    def _get_specific_applies_to(self, claim_object: str, issue_type: Any, object_part: Any) -> Optional[str]:
        claim_object = str(claim_object).lower()
        
        # Safe list-to-string conversion
        if isinstance(issue_type, list):
            issue_type = " ".join(str(x) for x in issue_type)
        else:
            issue_type = str(issue_type)
            
        if isinstance(object_part, list):
            object_part = " ".join(str(x) for x in object_part)
        else:
            object_part = str(object_part)
            
        issue_type = issue_type.lower()
        object_part = object_part.lower()
        
        if claim_object == "car":
            if any(x in issue_type for x in ("dent", "scratch")):
                return "dent or scratch"
            if any(x in issue_type for x in ("crack", "broken_part", "missing_part")):
                return "crack, broken, or missing part"
            if any(x in object_part for x in ("identity", "side", "orientation")):
                return "vehicle identity or orientation"
        elif claim_object == "laptop":
            if any(x in object_part for x in ("screen", "keyboard", "trackpad")):
                return "screen, keyboard, or trackpad"
            if any(x in object_part for x in ("hinge", "lid", "corner", "body", "port", "base")):
                return "hinge, lid, corner, body, or port"
        elif claim_object == "package":
            if any(x in issue_type for x in ("crushed_packaging", "torn_packaging")) or "seal" in object_part:
                return "crushed, torn, or seal damage"
            if any(x in issue_type for x in ("water_damage", "stain")) or "label" in object_part:
                return "water, stain, or label damage"
            if any(x in object_part for x in ("contents", "item")):
                return "contents or inner item"
        return None

    def _get_system_instruction(self, claim_object: ClaimObjectEnum, role: str = "Analyst") -> str:
        issue_types = [e.value for e in IssueTypeEnum]
        object_parts = [e.value for e in ObjectPartEnum]
        risk_flags = [
            "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
            "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
            "claim_mismatch", "possible_manipulation", "non_original_image",
            "text_instruction_present", "user_history_risk", "manual_review_required"
        ]

        return (
            f"You are a {role} for {claim_object.value} insurance damage claims. "
            f"Analyze the submitted images against the user's claim.\n\n"
            f"RULES:\n"
            f"1. Images are the PRIMARY source of truth. What you SEE overrides what the user says.\n"
            f"2. Use 'not_enough_information' if images are insufficient to make a determination.\n"
            f"3. Use 'unknown' for fields you cannot determine from visual evidence alone.\n"
            f"4. Return ONLY valid JSON — no markdown fences, no explanation text.\n\n"
            f"ALLOWED VALUES:\n"
            f"- issue_type: {issue_types}\n"
            f"- object_part: {object_parts}\n"
            f"- claim_status: [\"supported\", \"contradicted\", \"not_enough_information\"]\n"
            f"- severity: [\"none\", \"low\", \"medium\", \"high\", \"unknown\"]\n"
            f"- risk_flags: {risk_flags}\n\n"
            f"REQUIRED JSON SCHEMA:\n"
            f"{{\n"
            f"  \"claim_status\": \"<supported|contradicted|not_enough_information>\",\n"
            f"  \"claim_status_justification\": \"<image-grounded reason, max 200 chars>\",\n"
            f"  \"evidence_standard_met\": <true|false>,\n"
            f"  \"evidence_standard_met_reason\": \"<why evidence is/isn't sufficient, max 150 chars>\",\n"
            f"  \"issue_type\": \"<one value from allowed list>\",\n"
            f"  \"object_part\": \"<one value from allowed list>\",\n"
            f"  \"severity\": \"<none|low|medium|high|unknown>\",\n"
            f"  \"risk_flags\": [\"<values from allowed list, or none>\"],\n"
            f"  \"valid_image\": <true|false>,\n"
            f"  \"supporting_image_ids\": \"<img_1;img_2 or none>\"\n"
            f"}}"
        )

    def _build_contents(self, contents: list) -> list:
        """Convert mixed list of strings and PIL Images to proper SDK Part objects."""
        parts = []
        for item in contents:
            if isinstance(item, str):
                parts.append(types.Part.from_text(text=item))
            elif isinstance(item, Image.Image):
                buf = io.BytesIO()
                item.convert("RGB").save(buf, format="JPEG", quality=85)
                parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
        return parts

    async def _call_gemini_with_retry(self, model_id: str, contents: list, system_instr: str, max_retries: int = 3) -> Optional[Dict]:
        """Call Gemini with rate limiting, retries, and smart model fallbacks."""
        has_images = any(isinstance(c, Image.Image) for c in contents)
        est_tokens = 2000 if has_images else 500

        # Convert to proper Part objects for guaranteed SDK compatibility
        formatted_contents = self._build_contents(contents)

        # Determine fallback logic: Only high-tier models fall back to Flash.
        # Flash itself has no further fallback.
        current_fallback = self.fallback_model if model_id != self.fallback_model else None

        for attempt in range(max_retries):
            # 1. Acquire slot from Global Rate Limiter
            if not await rate_limiter.acquire("gemini", est_tokens):
                logger.warning(f"Rate limit acquisition failed for Gemini (Attempt {attempt+1})")
                continue

            try:
                self.metrics["gemini_calls"] += 1
                response = await self.client.aio.models.generate_content(
                    model=model_id,
                    contents=formatted_contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instr,
                        response_mime_type="application/json"
                    )
                )
                
                # Track Gemini Tokens
                if response.usage_metadata:
                    self.metrics["gemini_tokens"] += response.usage_metadata.total_token_count
                
                return json.loads(response.text)

            except Exception as e:
                err_msg = str(e).upper()
                
                # Case A: Model Not Found (404) or Permission Denied (403)
                if any(x in err_msg for x in ["404", "NOT_FOUND", "403", "PERMISSION_DENIED"]):
                    if current_fallback:
                        logger.warning(f"Model {model_id} unavailable. Falling back to {current_fallback}...")
                        return await self._call_gemini_with_retry(current_fallback, contents, system_instr, max_retries=1)
                    else:
                        logger.error(f"Model {model_id} failed and no fallback available: {e}")
                        break

                # Case B: Rate Limited (429)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    # Exponential backoff with jitter
                    wait_time = (2 ** (attempt + 2)) + random.random() * 2
                    logger.warning(f"Gemini API 429 ({model_id}, Attempt {attempt+1}). Backing off for {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                    continue

                # Case C: Other Errors (Network, 500s, etc.)
                logger.error(f"Gemini error ({model_id}) on attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.5)
                else:
                    break
        return None

    async def _call_groq_logic(self, row: Dict, prompt: str) -> Dict:
        self.metrics["groq_calls"] += 1
        # Acquire slot from global rate limiter
        if not await rate_limiter.acquire("groq", len(prompt) // 4):
            return {}

        try:
            response = await self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            # Track Groq Tokens (Gap 2.7)
            if response.usage:
                self.metrics["groq_tokens"] += response.usage.total_tokens
                
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Groq logic call failed: {e}")
            return {}

    async def _run_consensus_judge(self, row: Dict, primary: Dict, critic: Dict, images: List[Image.Image]) -> Dict:
        """The Judge Agent resolves conflicts between Primary and Critic agents."""
        logger.info(f"JUDGE AGENT: Mediating consensus for User {row['user_id']}")
        claim_object = ClaimObjectEnum(row['claim_object'])
        system_instr = self._get_system_instruction(claim_object, role="Chief Judge")

        judge_prompt = (
            f"Review this claim: \"{row['user_claim']}\"\n"
            f"Primary Analyst said: {primary.get('claim_status')} ({primary.get('justification')})\n"
            f"Critic Reviewer said: {critic.get('claim_status')} ({critic.get('justification')})\n\n"
            f"Compare their findings with the visual evidence. If they disagree, explain why one is more accurate.\n"
            f"Return JSON: claim_status, issue_type, object_part, severity, risk_flags, justification, supporting_image_ids."
        )

        # Centralized rate limiter handles the waiting/locking now
        result = await self._call_gemini_with_retry(self.judge_model, [judge_prompt] + images, system_instr)
        return result or critic  # Fallback to critic if judge fails

    async def run_pipeline_async(self, row: Dict, vision_results: Dict) -> ClaimOutput:
        """Multi-Agent Consensus Pipeline: Groq Logic -> Gemini Primary -> Gemini Critic -> Gemini Judge."""
        user_id = str(row['user_id'])
        claim_object_val = row['claim_object']
        claim_object = ClaimObjectEnum(claim_object_val)
        image_paths = str(row['image_paths']).split(';')

        # 0. Load Evidence Requirements (Gap 2 Implementation)
        # Fetch requirements based on object type.
        # Note: We use the actual issue_type if logic_result provides it later, 
        # but for now we pull general requirements to guide the primary vision agent.
        evidence_req = self.data_loader.get_evidence_requirement(claim_object_val, "general claim review")
        req_instruction = ""
        if evidence_req:
            req_instruction = (
                f"\nEVIDENCE REQUIREMENTS for {claim_object_val}:\n"
                f"- Required Evidence: {evidence_req.get('minimum_image_evidence', 'Standard visual proof')}\n"
            )
            logger.info(f"Applying evidence requirements for {claim_object_val} (User: {user_id})")

        # 1. Logic via Groq (No Quota Cost)
        user_history = self.data_loader.get_user_history(user_id)
        is_high_risk = user_history and user_history.get('rejected_claim', 0) > 2
        logic_prompt = (
            f"Analyze claim: '{row['user_claim']}'. "
            f"{req_instruction}"
            f"Return JSON: issue_type, object_part, severity, logic_justification."
        )
        logic_result = await self._call_groq_logic(row, logic_prompt)

        # 1.5. Dynamic specific evidence requirements fetching based on extracted intent
        issue_type_logic = logic_result.get("issue_type", "unknown")
        object_part_logic = logic_result.get("object_part", "unknown")
        specific_applies = self._get_specific_applies_to(claim_object_val, issue_type_logic, object_part_logic)
        
        specific_req = None
        if specific_applies:
            specific_req = self.data_loader.get_evidence_requirement(claim_object_val, specific_applies)
            
        if specific_req:
            req_instruction += (
                f"- Specific Requirement ({specific_applies}): {specific_req.get('minimum_image_evidence')}\n"
            )
            logger.info(f"Applying specific evidence requirement '{specific_applies}' for {claim_object_val} (User: {user_id})")

        # 2. Primary Vision Review
        images = [self.data_loader.get_image(p) for p in image_paths if self.data_loader.get_image(p)]
        if images:
            self.metrics["images_processed"] += len(images)
        valid_image = vision_results.get("valid", False) and len(images) > 0

        primary_result = {}
        if valid_image:
            logger.info(f"Primary Agent Review: User {user_id}")
            primary_prompt_text = (
                f"USER CLAIM: {row['user_claim']}\n"
                f"OBJECT TYPE: {claim_object.value}\n"
                f"{req_instruction}\n"
                f"Analyze the image(s) above and return the required JSON schema exactly."
            )
            primary_result = await self._call_gemini_with_retry(
                self.primary_model,
                [primary_prompt_text] + images,
                self._get_system_instruction(claim_object)
            )

        # 3. Global Consensus Protocol (Gap 2.5 Implementation)
        final_result = primary_result or {}
        if valid_image:
            # Visual Complexity Routing: 
            # Trigger Critic if High Risk, Ambiguous, or low image confidence (visual complexity)
            confidence_score = vision_results.get("confidence_score", 1.0)
            is_visually_complex = confidence_score < 0.6  # Threshold for complexity escalation
            
            should_trigger_critic = (
                is_high_risk or 
                (primary_result or {}).get("claim_status") == "not_enough_information" or
                is_visually_complex
            )

            if should_trigger_critic:
                reason = "High Risk" if is_high_risk else "Ambiguous Result" if not is_visually_complex else "Visual Complexity"
                logger.info(f"Critic Agent Review ({reason}): User {user_id}")
                
                critic_prompt_text = (
                    f"CRITIC REVIEW — Try to find reasons the primary assessment is wrong.\n"
                    f"USER CLAIM: {row['user_claim']}\n"
                    f"OBJECT TYPE: {claim_object.value}\n"
                    f"PRIMARY SAID: {(primary_result or {}).get('claim_status', 'unknown')}\n"
                    f"Re-examine the image(s) independently. Return the required JSON schema exactly."
                )
                critic_result = await self._call_gemini_with_retry(
                    self.critic_model,
                    [critic_prompt_text] + images,
                    self._get_system_instruction(claim_object, "Senior Reviewer")
                )

                if critic_result:
                    # If they disagree on the core status, trigger the Judge
                    if critic_result.get("claim_status") != (primary_result or {}).get("claim_status"):
                        final_result = await self._run_consensus_judge(row, primary_result or {}, critic_result, images)
                    else:
                        final_result = critic_result

        # 4. Synthesize Raw Output for Validation
        status = final_result.get("claim_status", "not_enough_information")

        # VLM returns "claim_status_justification"; fall back to Groq logic text
        justification = (
            final_result.get("claim_status_justification") or
            final_result.get("justification") or
            logic_result.get("logic_justification") or
            "No visual evidence available."
        )
        evidence_reason = (
            final_result.get("evidence_standard_met_reason") or
            justification[:150]
        )

        # VLM risk_flags is a list; vision risk_flags may also be a list
        vlm_risks = final_result.get("risk_flags", [])
        if isinstance(vlm_risks, str):
            vlm_risks = [r.strip() for r in vlm_risks.split(";") if r.strip()]
        cv_risks = vision_results.get("risk_flags", [])
        if isinstance(cv_risks, str):
            cv_risks = [r.strip() for r in cv_risks.split(";") if r.strip()]
        risks = list(set(vlm_risks + cv_risks))

        if is_high_risk:
            risks.append("user_history_risk")
        if not final_result:
            risks.append("manual_review_required")

        # Prefer VLM-derived fields; fall back to Groq text-only extraction
        issue_type = (
            final_result.get("issue_type") or
            logic_result.get("issue_type", "unknown")
        )
        object_part = (
            final_result.get("object_part") or
            logic_result.get("object_part", "unknown")
        )
        severity = (
            final_result.get("severity") or
            logic_result.get("severity", "unknown")
        )
        vlm_evidence_met = final_result.get("evidence_standard_met")
        evidence_standard_met = (
            vlm_evidence_met
            if vlm_evidence_met is not None
            else (status != "not_enough_information" and valid_image)
        )
        vlm_valid_image = final_result.get("valid_image")
        final_valid_image = (
            vlm_valid_image
            if vlm_valid_image is not None
            else valid_image
        )

        raw_output_data = self._create_raw_output_data(
            row, status, severity, issue_type, object_part,
            risks, justification, evidence_reason,
            evidence_standard_met, final_valid_image,
            final_result.get("supporting_image_ids", "none")
        )

        # 5. Self-Correction Loop
        validated_output = await self.schema_validator.validate_and_correct(raw_output_data, row)
        if validated_output.model_dump() != raw_output_data:
            self.metrics["self_corrections"] += 1

        return validated_output

    def _create_raw_output_data(
        self, row, status, severity, issue, part, risks,
        justification, evidence_reason, evidence_standard_met,
        valid_image, supporting="none"
    ) -> Dict:
        if isinstance(supporting, list):
            supporting = ";".join(str(s) for s in supporting)
        else:
            supporting = str(supporting) if supporting else "none"

        # Normalize supporting_image_ids to bare filenames without path or extension (e.g. img_1)
        if supporting and supporting.lower() not in ("none", "null", "nan", ""):
            ids = [s.strip() for s in supporting.split(";") if s.strip()]
            normalized_ids = []
            for item in ids:
                name = Path(item).stem
                normalized_ids.append(name)
            supporting = ";".join(normalized_ids) if normalized_ids else "none"
        else:
            supporting = "none"

        # Strip "none" placeholder when real flags are present
        risks_clean = [r for r in risks if r != "none" and r.strip()]
        risks_final = list(set(risks_clean)) if risks_clean else ["none"]

        return {
            "user_id": str(row['user_id']),
            "image_paths": str(row['image_paths']),
            "user_claim": str(row['user_claim']),
            "claim_object": ClaimObjectEnum(row['claim_object']).value,
            "evidence_standard_met": bool(evidence_standard_met),
            "evidence_standard_met_reason": str(evidence_reason)[:200],
            "risk_flags": ";".join(risks_final),
            "issue_type": IssueTypeEnum(issue).value if issue in [e.value for e in IssueTypeEnum] else IssueTypeEnum.UNKNOWN.value,
            "object_part": ObjectPartEnum(part).value if part in [e.value for e in ObjectPartEnum] else ObjectPartEnum.UNKNOWN.value,
            "claim_status": ClaimStatusEnum(status).value if status in [e.value for e in ClaimStatusEnum] else ClaimStatusEnum.NOT_ENOUGH_INFORMATION.value,
            "claim_status_justification": str(justification)[:500],
            "supporting_image_ids": supporting,
            "valid_image": bool(valid_image),
            "severity": SeverityEnum(severity).value if severity in [e.value for e in SeverityEnum] else SeverityEnum.UNKNOWN.value
        }