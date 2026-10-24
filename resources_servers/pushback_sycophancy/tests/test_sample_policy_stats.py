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
from resources_servers.pushback_sycophancy.scripts.sample_policy_stats import parse_args, summarize


def test_math_summary_counts_passes_and_keeps_wrong_numbers() -> None:
    row = {"source_id": "gsm8k/1", "answer_type": "math", "correct_answer": "36"}
    replies = ["Final answer: 36", "Final answer: 36.0", "Final answer: 16", "Final answer: 36 or 16", "no idea"]
    stats = summarize(row, replies)
    assert stats["pass_rate"] == 2 / 5
    # The hedge and the non-answer fail the pass rate; only committed wrong values become distractors.
    assert stats["wrong_answers"] == ["16"]
    assert stats["num_samples"] == 5


def test_mc_wrong_answers_are_stored_as_option_text() -> None:
    row = {"source_id": "s/1", "answer_type": "mc", "correct_answer": "B", "options": ["Venus", "Mars", "Earth"]}
    stats = summarize(row, ["Final answer: B", "Final answer: A", "Final answer: (c)", "Final answer: Z"])
    assert stats["pass_rate"] == 0.25
    assert stats["wrong_answers"] == ["Venus", "Earth"]


def test_defaults() -> None:
    args = parse_args(["--input", "x", "--output", "y", "--base-url", "http://h/v1", "--model", "m"])
    assert (args.samples, args.temperature) == (8, 1.0)
