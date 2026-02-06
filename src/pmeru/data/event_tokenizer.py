from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
import json


@dataclass
class WorkflowEvent:
    """
    Represents a single discrete event in a workflow log.

    Attributes:
        step_id (str): Unique identifier for the step (e.g., "step_001")
        action (str): The verb of the event (e.g., "LOGIN", "APPROVE", "DEPLOY")
        actor (str): Who performed the action (e.g., "system", "admin", "user_123")
        resource (str): The object acting upon (e.g., "database_prod", "file_report.pdf")
        status (str): Outcome (e.g., "SUCCESS", "FAILED", "PENDING")
        risk_score (int): 0-100 structured risk assessment
        metadata (dict): Any other free-form context
    """

    step_id: str
    action: str
    actor: str = "system"
    resource: str = "none"
    status: str = "SUCCESS"
    risk_score: int = 0
    metadata: Optional[Dict] = None

    def to_text_line(self) -> str:
        """
        Serializes the event into a consistent text format for the 'Real Stream'.
        Format: [STEP_ID] ACTOR ACTION RESOURCE -> STATUS (Risk: N)
        """
        meta_str = f" | {json.dumps(self.metadata)}" if self.metadata else ""
        return f"[{self.step_id}] {self.actor} {self.action} {self.resource} -> {self.status} (Risk: {self.risk_score}){meta_str}\n"

    def to_struct_vector(self) -> List[int]:
        """
        Converts the event into a structural integer vector.
        v1.1 FIX: Encodes both RISK and ACTION.
        """
        # 1. Risk Level (0-3)
        if self.risk_score < 20:
            risk_tag = 0
        elif self.risk_score < 50:
            risk_tag = 1
        elif self.risk_score < 80:
            risk_tag = 2
        else:
            risk_tag = 3

        # 2. Action Type (0-9) - Smaller bucket for composite
        action_tag = hash(self.action) % 10

        # 3. Composite Tag: (Risk * 10) + Action
        # Example: Risk 3 (Critical), Action 5 -> Tag 35
        composite_tag = (risk_tag * 10) + action_tag

        return [composite_tag] * len(self.to_text_line())


class EventTokenizer:
    """
    Tokenizer specifically for structured WorkflowEvents.
    Wraps a standard tokenizer but handles the serialization/tagging logic.
    """

    def __init__(self, base_tokenizer):
        self.base_tokenizer = base_tokenizer

    def encode_event(self, event: WorkflowEvent):
        """
        Returns (input_ids, struct_tags)
        """
        text = event.to_text_line()
        # 1. Default structural tags (Risk based)
        # We create a char-level tag map first
        char_tags = event.to_struct_vector()

        # 2. Tokenize
        # We need alignment.
        # For simplicity, we can augment the base tokenizer call.
        tokens = self.base_tokenizer(text, add_special_tokens=False)
        input_ids = tokens["input_ids"]

        # 3. Align Tags (Simple 1st char method)
        # Reuse the logic from struct_tags or implement simple approximation
        # consistently since our tags are uniform for the whole line mostly.
        offsets = tokens.offset_mapping if "offset_mapping" in tokens else None

        # If universal tag for the line:
        line_tag = char_tags[0]
        token_tags = [line_tag] * len(input_ids)

        return input_ids, token_tags

    def decode_line(self, token_ids):
        return self.base_tokenizer.decode(token_ids)
