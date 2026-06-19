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
        self.judge_model = os.getenv("GEMINI_JUDGE_MODEL", "gemini-1.5-pro")
        
        # Groq Fallback (Primary for Logic to save Gemini Quota)
        self.groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    def _get_system_instruction(self, claim_object: ClaimObjectEnum, role: str = "Analyst") -> str:
        issue_types = [e.value for e in IssueTypeEnum]
        object_parts = [e.value for e in ObjectPartEnum]
        
        return f"""You are an {role} for {claim_object.value} insurance claims.
Rules:
1. PRIMARY source: Visual evidence.
2. Allowed issues: {issue_types}
3. Allowed parts: {object_parts}
4. Be concise and professional.
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

    async def _run_consensus_judge(self, row: Dict, primary: Dict, critic: Dict, images: List[Image.Image]) -> Dict:
        """The Judge Agent resolves conflicts between Primary and Critic agents."""
        logger.info(f"JUDGE AGENT: Mediating consensus for User {row['user_id']}")
        claim_object = ClaimObjectEnum(row['claim_object'])
        system_instr = self._get_system_instruction(claim_object, role="Chief Judge")
        
        judge_prompt = f"""
        Review this claim: "{row['user_claim']}"
        Primary Analyst said: {primary.get('claim_status')} ({primary.get('justification')})
        Critic Reviewer said: {critic.get('claim_status')} ({critic.get('justification')})
        
        Compare their findings with the visual evidence. If they disagree, explain why one is more accurate.
        Return JSON: claim_status, issue_type, object_part, severity, risk_flags, justification, supporting_image_ids.
        """
        
        async with vision_lock:
            result = await self._call_gemini_with_retry(self.judge_model, [judge_prompt] + images, system_instr)
            await asyncio.sleep(1.0)
        return result or critic # Fallback to critic if judge fails

    async def run_pipeline_async(self, row: Dict, vision_results: Dict) -> ClaimOutput:
        """Multi-Agent Consensus Pipeline: Groq Logic -> Gemini Primary -> Gemini Critic -> Gemini Judge."""
        user_id = str(row['user_id'])
        claim_object = ClaimObjectEnum(row['claim_object'])
        image_paths = str(row['image_paths']).split(';')
        
        # 1. Logic via Groq (No Quota Cost)
        user_history = self.data_loader.get_user_history(user_id)
        is_high_risk = user_history and user_history.get('rejected_claim', 0) > 2
        logic_prompt = f"Analyze claim: '{row['user_claim']}'. Return JSON: issue_type, object_part, severity, logic_justification."
        logic_result = await self._call_groq_logic(row, logic_prompt)
        
        # 2. Primary Vision Review
        images = [self.data_loader.get_image(p) for p in image_paths if self.data_loader.get_image(p)]
        valid_image = vision_results.get("valid", False) and len(images) > 0
        
        primary_result = {}
        if valid_image:
            async with vision_lock:
                logger.info(f"Primary Agent Review: User {user_id}")
                primary_result = await self._call_gemini_with_retry(self.primary_model, [f"Verify claim: '{row['user_claim']}'"] + images, self._get_system_instruction(claim_object))
                await asyncio.sleep(1.0)

        # 3. Global Consensus Protocol
        final_result = primary_result
        if valid_image:
            # Trigger Critic if High Risk or Ambiguous
            if is_high_risk or primary_result.get("claim_status") == "not_enough_information":
                async with vision_lock:
                    logger.info(f"Critic Agent Review: User {user_id}")
                    critic_result = await self._call_gemini_with_retry(self.critic_model, [f"CRITIC REVIEW: Verify claim: '{row['user_claim']}'"] + images, self._get_system_instruction(claim_object, "Senior Reviewer"))
                    await asyncio.sleep(1.0)
                
                if critic_result:
                    # If they disagree on the core status, trigger the Judge
                    if critic_result.get("claim_status") != primary_result.get("claim_status"):
                        final_result = await self._run_consensus_judge(row, primary_result, critic_result, images)
                    else:
                        final_result = critic_result

        # 4. Synthesize Final Output
        status = final_result.get("claim_status", "not_enough_information")
        justification = final_result.get("justification", logic_result.get("logic_justification", "No visual evidence available."))
        risks = list(set(final_result.get("risk_flags", []) + vision_results.get("risk_flags", [])))
        if is_high_risk: risks.append("user_history_risk")
        if not final_result: risks.append("vision_service_unavailable")

        return self._create_output(
            row, status,
            logic_result.get("severity", "unknown"),
            logic_result.get("issue_type", "unknown"),
            logic_result.get("object_part", "unknown"),
            risks, justification, valid_image,
            final_result.get("supporting_image_ids", "none")
        )

    def _create_output(self, row, status, severity, issue, part, risks, justification, valid, supporting="none") -> ClaimOutput:
        if isinstance(supporting, list): supporting = ";".join(str(s) for s in supporting)
        elif supporting is None: supporting = "none"
        else: supporting = str(supporting)
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
