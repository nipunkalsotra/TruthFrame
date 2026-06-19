import os
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

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global lock for Gemini Vision to prevent concurrent rate limit hits on the Free Tier
vision_lock = asyncio.Lock()

class ReasoningEngine:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        
        # Tiered Models
        self.primary_model = os.getenv("GEMINI_TEXT_MODEL", "gemini-1.5-flash")
        self.critic_model = os.getenv("GEMINI_CRITIC_MODEL", "gemini-1.5-pro")
        
        # Groq Fallback (Primary for Logic to save Gemini Quota)
        self.groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    def _get_system_instruction(self, claim_object: ClaimObjectEnum) -> str:
        issue_types = [e.value for e in IssueTypeEnum]
        object_parts = [e.value for e in ObjectPartEnum]
        
        return f"""You are an elite Multi-Modal Evidence Review System for {claim_object.value} insurance claims.
Rules:
1. PRIMARY source: Visual evidence.
2. DIFFERENT object (e.g. keyboard/mouse when claim is CAR) -> 'contradicted' + 'wrong_object' risk.
3. Allowed issues: {issue_types}
4. Allowed parts: {object_parts}
5. Be concise and professional.
"""

    async def _call_gemini_with_retry(self, model_id: str, contents: list, system_instr: str, max_retries: int = 3) -> Optional[Dict]:
        """Async Orchestrator: Handles Quotas with Exponential Backoff and Jitter."""
        for attempt in range(max_retries):
            try:
                response = await self.client.aio.models.generate_content(
                    model=model_id,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instr,
                        response_mime_type="application/json"
                    )
                )
                return json.loads(response.text)
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait_time = (2 ** attempt) + random.random() * 2
                    logger.warning(f"Rate Limit hit for {model_id}. Retrying in {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Gemini error ({model_id}): {e}")
                    break
        return None

    async def _call_groq_logic(self, row: Dict, prompt: str) -> Dict:
        """High-speed logic extraction via Groq to save Gemini Vision quota."""
        try:
            response = await self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Groq logic call failed: {e}")
            return {}

    async def run_pipeline_async(self, row: Dict, vision_results: Dict) -> ClaimOutput:
        """Quota-Friendly reasoning pipeline: Groq for logic, Sequential Gemini for Vision."""
        user_id = str(row['user_id'])
        claim_object = ClaimObjectEnum(row['claim_object'])
        image_paths = str(row['image_paths']).split(';')
        
        # 1. Risk Assessment & Logic (Via Groq - No Gemini Quota Cost)
        user_history = self.data_loader.get_user_history(user_id)
        is_high_risk = user_history and user_history.get('rejected_claim', 0) > 2
        
        logic_prompt = f"""Analyze this claim for {claim_object.value}: "{row['user_claim']}"
        Return JSON with fields: issue_type, object_part, severity, logic_justification.
        Allowed issue_types: {[e.value for e in IssueTypeEnum]}
        Allowed object_parts: {[e.value for e in ObjectPartEnum]}"""
        
        logic_result = await self._call_groq_logic(row, logic_prompt)
        
        # 2. Vision Verification (Via Gemini - Sequential to avoid 429)
        images = [self.data_loader.get_image(p) for p in image_paths if self.data_loader.get_image(p)]
        valid_image = vision_results.get("valid", False) and len(images) > 0
        
        vision_result = {}
        if valid_image:
            # Use a lock to ensure only one Gemini Vision call happens at a time globally
            async with vision_lock:
                logger.info(f"Lock acquired for Gemini Vision: User {user_id}")
                system_instr = self._get_system_instruction(claim_object)
                vision_prompt = f"Verify if images support claim: '{row['user_claim']}'. Return JSON: claim_status, risk_flags, justification, supporting_image_ids."
                vision_result = await self._call_gemini_with_retry(self.primary_model, [vision_prompt] + images, system_instr)
                # Small delay to let the API 'breathe'
                await asyncio.sleep(1.0)

        # 3. Critic Check (Only for High Risk)
        if is_high_risk and vision_result:
            async with vision_lock:
                logger.info(f"Triggering Critic Agent (Sequential): User {user_id}")
                critic_result = await self._call_gemini_with_retry(self.critic_model, ["CRITIC REVIEW: Verify this claim thoroughly."] + images, self._get_system_instruction(claim_object))
                if critic_result: vision_result = critic_result

        # 4. Synthesize Final Output
        status = vision_result.get("claim_status", "not_enough_information")
        justification = vision_result.get("justification", logic_result.get("logic_justification", "No visual evidence available."))
        risks = list(set(vision_result.get("risk_flags", []) + vision_results.get("risk_flags", [])))
        if is_high_risk: risks.append("user_history_risk")
        if not vision_result: risks.append("vision_service_unavailable")

        return self._create_output(
            row, 
            status,
            logic_result.get("severity", "unknown"),
            logic_result.get("issue_type", "unknown"),
            logic_result.get("object_part", "unknown"),
            risks,
            justification,
            valid_image,
            vision_result.get("supporting_image_ids", "none")
        )

    def _create_output(self, row, status, severity, issue, part, risks, justification, valid, supporting="none") -> ClaimOutput:
        if isinstance(supporting, list):
            supporting = ";".join(str(s) for s in supporting)
        elif supporting is None:
            supporting = "none"
        else:
            supporting = str(supporting)

        if not risks or risks == ["none"]: risks = ["none"]
        
        return ClaimOutput(
            user_id=str(row['user_id']),
            image_paths=str(row['image_paths']),
            user_claim=row['user_claim'],
            claim_object=ClaimObjectEnum(row['claim_object']),
            evidence_standard_met=(status != "not_enough_information" and valid),
            evidence_standard_met_reason=justification[:200],
            risk_flags=";".join(risks),
            issue_type=IssueTypeEnum(issue) if issue in [e.value for e in IssueTypeEnum] else IssueTypeEnum.UNKNOWN,
            object_part=ObjectPartEnum(part) if part in [e.value for e in ObjectPartEnum] else ObjectPartEnum.UNKNOWN,
            claim_status=ClaimStatusEnum(status) if status in [e.value for e in ClaimStatusEnum] else ClaimStatusEnum.NOT_ENOUGH_INFORMATION,
            claim_status_justification=justification[:500],
            supporting_image_ids=supporting,
            valid_image=valid,
            severity=SeverityEnum(severity) if severity in [e.value for e in SeverityEnum] else SeverityEnum.UNKNOWN
        )
