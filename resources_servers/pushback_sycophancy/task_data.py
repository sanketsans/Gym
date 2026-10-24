# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the pushback_sycophancy server.

A gymnasium-style environment: rows reach ``step()`` as top-level extras. Two modes:

- ``prefill`` (default): the prompt in ``responses_create_params.input`` already holds
  the question, a prefilled first answer, and the first pushback; ``followups`` holds
  the later, escalating pushbacks. The row fixes the arm.
- ``live``: the prompt holds only the question. The policy answers it, and that answer
  picks the arm (correct -> hold, wrong -> update). One fixed pushback follows, naming
  ``distractor`` on the hold arm and ``correct_answer`` on the update arm. MC only.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# Pushbacks per episode, counting the one already in the prompt.
MAX_PUSHBACKS = 5


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    mode: Literal["prefill", "live"] = Field(
        default="prefill",
        description="'prefill': the row fixes the first answer and the arm. 'live': the policy's first answer does.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    arm: Optional[Literal["hold", "update"]] = Field(
        default=None,
        description=(
            "Prefill mode only. 'hold': the prefilled answer is correct and the user pushes a wrong one. "
            "'update': the prefilled answer is wrong and the user pushes the correct one."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    answer_type: Literal["mc", "math", "text"] = Field(
        description="How answers are compared: option letter, math equivalence, or normalised text.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    correct_answer: str = Field(
        description="Ground-truth answer.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    user_claim: Optional[str] = Field(
        default=None,
        description=(
            "Prefill mode only. The answer the simulated user argues for "
            "(wrong on the hold arm, correct on the update arm)."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    prefilled_answer: Optional[str] = Field(
        default=None,
        description="Prefill mode only. The answer committed to in the prefilled assistant turn.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    distractor: Optional[str] = Field(
        default=None,
        description="Live mode only. The wrong option the user pushes when the policy's first answer is correct.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    followups: List[str] = Field(
        default_factory=list,
        max_length=MAX_PUSHBACKS - 1,
        description=(
            "Pushback turns 2..T (T <= 5), sent after each policy reply. Empty for a single-pushback episode."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="Accepted alternative spellings of correct_answer (text answers).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    pressure: List[str] = Field(
        default_factory=list,
        description="Pressure type of each pushback turn (P1-P7, U-E, U-B).",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    source: Optional[str] = Field(
        default=None,
        description="Seed dataset the question came from.",
        json_schema_extra={"consumed_by": ["metrics", "provenance"]},
    )
    source_id: Optional[str] = Field(
        default=None,
        description="Identifier of the question within its source dataset.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    options: Optional[List[str]] = Field(
        default=None,
        description="Multiple-choice option texts, in the order they were shown (mc answers).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )

    @model_validator(mode="after")
    def _check_mode_fields(self) -> "TaskData":
        if self.mode == "prefill":
            missing = [name for name in ("arm", "user_claim", "prefilled_answer") if getattr(self, name) is None]
            if missing:
                raise ValueError(f"prefill rows need {missing}")
            return self
        if self.answer_type != "mc":
            raise ValueError("live rows are multiple-choice only (answer_type 'mc')")
        if not self.distractor:
            raise ValueError("live rows need a distractor")
        if self.distractor.strip().upper() == self.correct_answer.strip().upper():
            raise ValueError("the live-mode distractor must differ from correct_answer")
        if self.arm is not None or self.followups:
            raise ValueError("live rows pick their arm at runtime and have exactly one pushback")
        return self
