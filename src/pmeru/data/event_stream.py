from dataclasses import dataclass, field
from typing import List, Optional
import torch
from torch.utils.data import IterableDataset
import json
import logging
import random
from .event_tokenizer import WorkflowEvent, EventTokenizer

logger = logging.getLogger(__name__)


@dataclass
class SyntheticEventConfig:
    num_events: int = 10000
    verbs: List[str] = field(
        default_factory=lambda: [
            "LOGIN",
            "LOGOUT",
            "DEPLOY",
            "ROLLBACK",
            "APPROVE",
            "DENY",
            "SCALE_UP",
            "HEARTBEAT",
        ]
    )
    actors: List[str] = field(
        default_factory=lambda: [
            "user_alice",
            "user_bob",
            "system_cron",
            "admin_root",
            "service_payment",
        ]
    )
    resources: List[str] = field(
        default_factory=lambda: [
            "auth_db",
            "web_cluster",
            "payment_gateway",
            "cdn_cache",
            "audit_log",
        ]
    )
    risk_probs: List[float] = field(
        default_factory=lambda: [0.7, 0.2, 0.09, 0.01]
    )  # Low, Med, High, Critical


class EventStreamDataset(IterableDataset):
    """
    Simulates a stream of enterprise workflow events.
    """

    def __init__(
        self, tokenizer, config: SyntheticEventConfig, seq_len=1024, infinite=False
    ):
        self.tokenizer = tokenizer
        self.event_tokenizer = EventTokenizer(tokenizer)
        self.config = config
        self.seq_len = seq_len
        self.infinite = infinite

    def _generate_event(self, step_idx):
        action = random.choice(self.config.verbs)
        actor = random.choice(self.config.actors)
        resource = random.choice(self.config.resources)

        # Determine metrics based on simple logic for pattern learning
        if action in ["DEPLOY", "ROLLBACK"]:
            risk = random.randint(30, 90)  # Riskier actions
        elif action in ["LOGIN"]:
            risk = random.randint(0, 30)
        else:
            # Weighted random risk
            risk = random.choices([10, 40, 70, 95], weights=self.config.risk_probs)[0]

        status = "SUCCESS"
        if risk > 80:
            status = random.choice(["FAILED", "DENIED", "PENDING"])

        event = WorkflowEvent(
            step_id=f"evt_{step_idx:08d}",
            action=action,
            actor=actor,
            resource=resource,
            status=status,
            risk_score=risk,
            metadata={"timestamp": int(100000 + step_idx)},
        )
        return event

    def __iter__(self):
        buffer_tokens = []
        buffer_tags = []

        step_idx = 0
        while True:
            # Generate Event
            event = self._generate_event(step_idx)
            step_idx += 1

            # Tokenize & Tag
            input_ids, tags = self.event_tokenizer.encode_event(event)

            buffer_tokens.extend(input_ids)
            buffer_tags.extend(tags)

            # Yield Batches
            while len(buffer_tokens) >= self.seq_len:
                yield {
                    "input_ids": torch.tensor(
                        buffer_tokens[: self.seq_len], dtype=torch.long
                    ),
                    "attention_mask": torch.ones(self.seq_len, dtype=torch.bool),
                    "struct_tags": torch.tensor(
                        buffer_tags[: self.seq_len], dtype=torch.long
                    ),
                    "labels": torch.tensor(
                        buffer_tokens[: self.seq_len], dtype=torch.long
                    ),
                }

                buffer_tokens = buffer_tokens[self.seq_len :]
                buffer_tags = buffer_tags[self.seq_len :]

            if not self.infinite and step_idx >= self.config.num_events:
                break
