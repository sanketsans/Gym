# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the pushback_sycophancy server.

A gymnasium-style environment: rows reach ``step()`` as top-level extras. The prompt in
``responses_create_params.input`` holds only the question. The policy answers it, and that
answer picks the arm (correct -> hold, wrong -> update). One pushback follows, naming
``distractor`` on the hold arm and ``correct_answer`` on the update arm. MC or math.
"""

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _canonical(answer: str) -> str:
    return answer.strip().upper().replace(",", "").replace("$", "")


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    answer_type: Literal["mc", "math"] = Field(
        description="How answers are compared: option letter, or math equivalence.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    correct_answer: str = Field(
        description="Ground-truth answer.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    distractor: str = Field(
        description="The wrong answer the user pushes when the policy's first answer is correct.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    distractor_source: Optional[Literal["policy", "model", "option", "perturbation", "generated"]] = Field(
        default=None,
        description=(
            "Where the distractor came from: the base policy's or a profiling model's most frequent wrong answer, "
            "a wrong MC option, a numeric perturbation, or a model-written plausible wrong answer."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    pushbacks: Optional[Dict[Literal["hold", "update"], Union[str, Dict[str, str]]]] = Field(
        default=None,
        description=(
            "Frozen datasets: the exact pushback for each arm, used instead of the template or a simulated user "
            "so that every model on the same arm sees the same text. Either one text per arm, or {style: text} "
            "per arm, from which the server picks a style per episode (pushback_styles)."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    pushback_variant: Optional[str] = Field(
        default=None,
        description="Style label of a single-text frozen pushback (for example 'bare' or 'argued'), for metrics.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="Accepted alternative spellings of correct_answer.",
        json_schema_extra={"consumed_by": ["verify"]},
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
        json_schema_extra={"consumed_by": ["verify"]},
    )

    @model_validator(mode="after")
    def _check(self) -> "TaskData":
        # A cheap string check; data builders should also reject distractors that are merely equivalent (0.5 vs 1/2).
        if _canonical(self.distractor) == _canonical(self.correct_answer):
            raise ValueError("the distractor must differ from correct_answer")
        if self.pushbacks is not None:
            if set(self.pushbacks) != {"hold", "update"}:
                raise ValueError("frozen pushbacks need both a 'hold' and an 'update' text")
            hold, update = self.pushbacks["hold"], self.pushbacks["update"]
            if isinstance(hold, dict) != isinstance(update, dict):
                raise ValueError("frozen pushbacks must be a text on both arms or a {style: text} map on both")
            if isinstance(hold, dict) and (not hold or set(hold) != set(update)):
                raise ValueError("styled pushbacks need the same non-empty set of styles on both arms")
        return self
