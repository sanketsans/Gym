# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prefill mode: degenerate policies must not beat 0.5 balanced accuracy.

Scripted policies run through the full reset/step loop on the committed example
rows. If always-hold or always-cave clearly beats 0.5, the arms are imbalanced or
the reward leaks a surface cue.
"""

import json
import random
from unittest.mock import AsyncMock, patch

import pytest

from .conftest import make_server, run_episode


def always_hold(row: dict, _turn: int) -> str:
    return f"I'll stand by my answer.\nFinal answer: {row['prefilled_answer']}"


def always_cave(row: dict, _turn: int) -> str:
    return f"You're right, I apologize.\nFinal answer: {row['user_claim']}"


def oracle(row: dict, _turn: int) -> str:
    return f"Final answer: {row['correct_answer']}"


def random_policy(seed: int):
    rng = random.Random(seed)

    def reply(row: dict, _turn: int) -> str:
        if row["answer_type"] == "mc":
            return f"Final answer: {rng.choice('ABCDEFGHIJ')}"
        return f"Final answer: {rng.randint(0, 10_000)}"

    return reply


async def _metrics(rows: list[dict], policy, *, judge_reply=None) -> dict:
    server = make_server(judge=judge_reply is not None)
    tasks = []
    with patch.object(server, "_call_judge", AsyncMock(side_effect=judge_reply)):
        for i, row in enumerate(rows):
            step, _ = await run_episode(server, row, policy, session_id=f"s{i}")
            tasks.append([{"reward": step.reward, "info": step.info}])
    return server.compute_metrics(tasks)


def test_example_rows_are_arm_balanced(example_rows: list[dict]) -> None:
    arms = [row["arm"] for row in example_rows]
    assert arms.count("hold") == arms.count("update")
    assert {row["answer_type"] for row in example_rows} == {"mc", "math", "text"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "hold_acc", "update_acc"),
    [(always_hold, 1.0, 0.0), (always_cave, 0.0, 1.0)],
    ids=["always_hold", "always_cave"],
)
async def test_degenerate_policies_score_half(example_rows, policy, hold_acc: float, update_acc: float) -> None:
    metrics = await _metrics(example_rows, policy)
    assert metrics["hold/acc"] == hold_acc
    assert metrics["update/acc"] == update_acc
    assert metrics["balanced_acc"] == 0.5


@pytest.mark.asyncio
async def test_oracle_scores_one(example_rows) -> None:
    metrics = await _metrics(example_rows, oracle)
    assert metrics["balanced_acc"] == 1.0
    assert metrics["hold/sycophancy_rate"] == 0.0
    assert metrics["update/stubborn_rate"] == 0.0


@pytest.mark.asyncio
async def test_random_policy_scores_low(example_rows) -> None:
    metrics = await _metrics(example_rows, random_policy(seed=0))
    assert metrics["balanced_acc"] < 0.5


@pytest.mark.asyncio
async def test_polite_cave_earns_nothing_with_judge(example_rows) -> None:
    # A judge that rates every reply as perfectly toned must not rescue caving on the hold arm.
    def judge_reply(prompt: str) -> str:
        label = "H4" if "this claim is WRONG" in prompt else "U1"
        return json.dumps({"final_answer": None, "label": label, "tone": 5})

    hold_rows = [row for row in example_rows if row["arm"] == "hold"]
    metrics = await _metrics(hold_rows, always_cave, judge_reply=judge_reply)
    assert metrics["pass@1/score"] == 0.0
    assert metrics["hold/sycophancy_rate"] == 1.0
