import os
import json
import time
import random
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai import types
from groq import Groq
from PIL import Image

from models import (
    ClaimObjectEnum, IssueTypeEnum, ObjectPartEnum, 
    ClaimStatusEnum, SeverityEnum, ClaimOutput
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReasoningEngine:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        
        # Tiered Models
        self.primary_model = os.getenv("GEMINI_TEXT_MODEL", "gemini-1.5-flash")
        self.critic_model = os.getenv("GEMINI_CRITIC_MODEL", "gemini-1.5-pro")
        
        # Groq Fallback (Critical for Demo Uptime)
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
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

    def _call_gemini_with_retry(self, model_id: str, contents: list, system_instr: str, max_retries: int = 3) -> Optional[Dict]:
        """Orchestrator: Handles Quotas with Exponential Backoff and Jitter."""
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
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
                    wait_time = (2 ** attempt) + random.random()
                    logger.warning(f"Rate Limit hit for {model_id}. Retrying in {wait_time:.2f}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Gemini error ({model_id}): {e}")
                    break
        return None

    def _call_groq_fallback(self, row: Dict, prompt: str) -> Dict:
        """Fallback: Switch to Groq if Gemini is completely exhausted."""
        try:
            logger.info("CRITICAL: Switching to Groq Fallback...")
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Groq fallback failed: {e}")
            return {"claim_status": "not_enough_information", "justification": "All AI services exhausted."}

    def analyze_claim_multimodal(self, row: Dict, images: List[Image.Image], is_critic: bool = False) -> Dict:
        claim_object = ClaimObjectEnum(row['claim_object'])
        model_id = self.critic_model if is_critic else self.primary_model
        system_instr = self._get_system_instruction(claim_object)
        
        prompt = f"""Review claim: "{row['user_claim']}" for {claim_object.value}. 
Return JSON: claim_status, issue_type, object_part, severity, risk_flags, justification, supporting_image_ids."""
        
        # 1. Try Gemini (Multimodal)
        result = self._call_gemini_with_retry(model_id, [prompt] + images, system_instr)
        
        # 2. If Gemini fails, use Groq (Textual reasoning fallback)
        if not result and not is_critic:
            result = self._call_groq_fallback(row, prompt + " (Note: Visuals unavailable, use text logic)")
            result["risk_flags"] = result.get("risk_flags", []) + ["vision_service_exhausted"]
            
        return result or {"claim_status": "not_enough_information", "justification": "AI service unavailable."}

    def run_pipeline(self, row: Dict, vision_results: Dict) -> ClaimOutput:
        """Main reasoning pipeline with Risk Routing and Multi-Agent Verification."""
        user_id = str(row['user_id'])
        claim_object = ClaimObjectEnum(row['claim_object'])
        image_paths = row['image_paths'].split(';')
        
        # 1. Risk Assessment Router
        user_history = self.data_loader.get_user_history(user_id)
        is_high_risk = user_history and user_history.get('rejected_claim', 0) > 2
        initial_risk_flags = vision_results.get("risk_flags", [])
        if is_high_risk: initial_risk_flags.append("user_history_risk")
            
        # 2. Load Images
        images = [self.data_loader.get_image(p) for p in image_paths if self.data_loader.get_image(p)]
        valid_image = vision_results.get("valid", False) and len(images) > 0
        
        if not valid_image:
            return self._create_output(row, "not_enough_information", "none", "unknown", "unknown", initial_risk_flags, "Invalid or missing images.", False)

        # 3. Primary Agent Analysis
        result = self.analyze_claim_multimodal(row, images, is_critic=False)
        
        # 4. Multi-Agent Cross-Verification (Critic)
        if (is_high_risk or result.get("claim_status") == "not_enough_information") and "exhausted" not in str(result):
            logger.info(f"Triggering Critic Agent for user {user_id}")
            critic_result = self.analyze_claim_multimodal(row, images, is_critic=True)
            if critic_result and critic_result.get("claim_status") != "not_enough_information":
                result = critic_result

        final_risks = list(set(result.get("risk_flags", []) + initial_risk_flags))
        if "none" in final_risks and len(final_risks) > 1: final_risks.remove("none")

        return self._create_output(
            row, 
            result.get("claim_status", "not_enough_information"),
            result.get("severity", "unknown"),
            result.get("issue_type", "unknown"),
            result.get("object_part", "unknown"),
            final_risks,
            result.get("justification", "No justification provided."),
            valid_image,
            result.get("supporting_image_ids", "none")
        )

    def _create_output(self, row, status, severity, issue, part, risks, justification, valid, supporting="none") -> ClaimOutput:
        """Enforces schema compliance and fixes formatting errors."""
        # ENSURE supporting_image_ids is a string (Fix for Pydantic error)
        if isinstance(supporting, list):
            supporting = ";".join(str(s) for s in supporting)
        elif supporting is None:
            supporting = "none"
        else:
            supporting = str(supporting)

        if not risks: risks = ["none"]
        
        return ClaimOutput(
            user_id=str(row['user_id']),
            image_paths=row['image_paths'],
            user_claim=row['user_claim'],
            claim_object=ClaimObjectEnum(row['claim_object']),
            evidence_standard_met=(status != "not_enough_information" and valid),
            evidence_standard_met_reason=justification[:200],
            risk_flags=";".join(risks),
            issue_type=IssueTypeEnum(issue) if issue in [e.value for e in IssueTypeEnum] else IssueTypeEnum.UNKNOWN,
            object_part=ObjectPartEnum(part) if part in [e.value for e in ObjectPartEnum] else ObjectPartEnum.UNKNOWN,
            claim_status=ClaimStatusEnum(status) if status in [e.value for e in ClaimStatusEnum] else ClaimStatusEnum.NOT_ENOUGH_INFORMATION,
            claim_status_justification=justification[:500],
            supporting_image_ids=supporting if supporting else "none",
            valid_image=valid,
            severity=SeverityEnum(severity) if severity in [e.value for e in SeverityEnum] else SeverityEnum.UNKNOWN
        )
