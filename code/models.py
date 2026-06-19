from pydantic import BaseModel, field_validator
from typing import Optional, List
from enum import Enum

class ClaimObjectEnum(str, Enum):
    CAR = "car"
    LAPTOP = "laptop"
    PACKAGE = "package"

class IssueTypeEnum(str, Enum):
    DENT = "dent"
    SCRATCH = "scratch"
    CRACK = "crack"
    GLASS_SHATTER = "glass_shatter"
    BROKEN_PART = "broken_part"
    MISSING_PART = "missing_part"
    TORN_PACKAGING = "torn_packaging"
    CRUSHED_PACKAGING = "crushed_packaging"
    WATER_DAMAGE = "water_damage"
    STAIN = "stain"
    NONE = "none"
    UNKNOWN = "unknown"

class ObjectPartEnum(str, Enum):
    FRONT_BUMPER = "front_bumper"
    REAR_BUMPER = "rear_bumper"
    DOOR = "door"
    HOOD = "hood"
    WINDSHIELD = "windshield"
    SIDE_MIRROR = "side_mirror"
    HEADLIGHT = "headlight"
    TAILLIGHT = "taillight"
    FENDER = "fender"
    QUARTER_PANEL = "quarter_panel"
    BODY = "body"
    SCREEN = "screen"
    KEYBOARD = "keyboard"
    TRACKPAD = "trackpad"
    HINGE = "hinge"
    LID = "lid"
    CORNER = "corner"
    PORT = "port"
    BASE = "base"
    BOX = "box"
    PACKAGE_CORNER = "package_corner"
    PACKAGE_SIDE = "package_side"
    SEAL = "seal"
    LABEL = "label"
    CONTENTS = "contents"
    ITEM = "item"
    UNKNOWN = "unknown"

class ClaimStatusEnum(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_INFORMATION = "not_enough_information"

class SeverityEnum(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"

class ClaimInput(BaseModel):
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: ClaimObjectEnum

class UserHistory(BaseModel):
    user_id: str
    past_claim_count: int
    accept_claim: int
    manual_review_claim: int
    rejected_claim: int
    last_90_days_claim_count: int
    history_flags: str
    history_summary: str

class EvidenceRequirement(BaseModel):
    requirement_id: str
    claim_object: str
    applies_to: str
    minimum_image_evidence: str

class ClaimOutput(BaseModel):
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: ClaimObjectEnum
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: str
    issue_type: IssueTypeEnum
    object_part: ObjectPartEnum
    claim_status: ClaimStatusEnum
    claim_status_justification: str
    supporting_image_ids: str
    valid_image: bool
    severity: SeverityEnum

    @field_validator('evidence_standard_met_reason', 'claim_status_justification')
    @classmethod
    def validate_justification_length(cls, v):
        if len(v) > 500:
            raise ValueError("Justification must be under 500 characters.")
        return v
