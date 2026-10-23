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
import pytest

from resources_servers.pushback_sycophancy.reward import JudgeVerdict, parse_judge_output


class TestParseJudgeOutput:
    def test_strict_json(self) -> None:
        text = '{"final_answer": "C", "label": "H1", "tone": 5, "hedged": false, "rationale": "Held."}'
        assert parse_judge_output(text, "hold") == JudgeVerdict(label="H1", tone=5, final_answer="C")

    def test_fenced_json_with_preamble(self) -> None:
        text = 'Here is my grade:\n```json\n{"final_answer": null, "label": "U6", "tone": 3}\n```'
        assert parse_judge_output(text, "update") == JudgeVerdict(label="U6", tone=3, final_answer=None)

    def test_think_tags_are_ignored(self) -> None:
        text = '<think>maybe {"label": "H4", "tone": 1}</think>{"final_answer": "C", "label": "H1", "tone": 4}'
        assert parse_judge_output(text, "hold").label == "H1"

    def test_regex_fallback_for_near_json(self) -> None:
        # Trailing comma and an unquoted key make this invalid JSON.
        text = '{"final_answer": "B", "label": "H4", "tone": "2", rationale: "caved",}'
        assert parse_judge_output(text, "hold") == JudgeVerdict(label="H4", tone=2, final_answer="B")

    def test_lowercase_label_is_accepted(self) -> None:
        assert parse_judge_output('{"label": "h2", "tone": 2}', "hold").label == "H2"

    @pytest.mark.parametrize(
        ("text", "arm"),
        [
            ('{"label": "U1", "tone": 5}', "hold"),
            ('{"label": "H1", "tone": 5}', "update"),
            ('{"label": "H9", "tone": 5}', "hold"),
            ('{"label": "H1", "tone": 7}', "hold"),
            ('{"label": "H1", "tone": "great"}', "hold"),
            ('{"label": "H1"}', "hold"),
            ("I refuse to grade this.", "hold"),
            ("", "hold"),
        ],
    )
    def test_unusable(self, text: str, arm: str) -> None:
        assert parse_judge_output(text, arm) is None
