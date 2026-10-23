# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the pushback_sycophancy server.

A gymnasium-style environment: rows reach ``step()`` as top-level extras. The
prompt in ``responses_create_params.input`` already holds the question, the
prefilled first answer, and the first pushback; ``followups`` holds the later,
escalating pushbacks the environment replays after each policy turn.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    arm: Literal["hold", "update"] = Field(
        description=(
            "'hold': the prefilled answer is correct and the user pushes a wrong one. "
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
    user_claim: str = Field(
        description="The answer the simulated user argues for (wrong on the hold arm, correct on the update arm).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    prefilled_answer: str = Field(
        description="The answer committed to in the prefilled assistant turn.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    followups: List[str] = Field(
        default_factory=list,
        description="Pushback turns 2..T, sent after each policy reply. Empty for a single-pushback episode.",
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
