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
import collections
import json
import random
from pathlib import Path

import pytest
import yaml

from resources_servers.pushback_sycophancy.scripts.build_seed import (
    Decontaminator,
    Seed,
    _mc_letter,
    _mc_seed,
    _numeric_distractors,
    apply_policy_stats,
    assign_arms,
    build_live_row,
    build_row,
    in_pass_rate_band,
    is_validation,
    parse_args,
    pick_pressures,
)
from resources_servers.pushback_sycophancy.task_data import TaskData


TEMPLATES = yaml.safe_load((Path(__file__).parents[1] / "prompts" / "pushback_templates.yaml").read_text())


def _seed(**overrides) -> Seed:
    base = dict(
        source="unit",
        source_id="unit/1",
        question="2 + 2 = ?",
        answer_type="math",
        correct_answer="4",
        distractors=["5"],
    )
    return Seed(**(base | overrides))


class TestTemplates:
    def test_every_template_fills_with_a_claim(self) -> None:
        for code, templates in TEMPLATES.items():
            assert len(templates) >= 10, code
            for template in templates:
                assert "{claim}" in template or code == "P1", (code, template)
                template.format(claim="X")


class TestMcSeed:
    def test_shuffles_and_tracks_the_correct_letter(self) -> None:
        options = ["right", "w1", "w2", "w3"]
        letters = collections.Counter(
            _mc_seed("s", f"s/{i}", "Q?", options, 0, random.Random(i)).correct_answer for i in range(200)
        )
        assert set(letters) == {"A", "B", "C", "D"}
        seed = _mc_seed("s", "s/0", "Q?", options, 0, random.Random(0))
        assert seed.options[ord(seed.correct_answer) - ord("A")] == "right"
        assert seed.correct_answer not in seed.distractors
        assert f"{seed.correct_answer}. right" in seed.question

    def test_caps_options_at_ten_and_keeps_the_correct_one(self) -> None:
        options = [f"o{i}" for i in range(14)]
        seed = _mc_seed("s", "s/0", "Q?", options, 13, random.Random(0))
        assert len(seed.options) == 10
        assert "o13" in seed.options


def test_numeric_distractors_are_wrong_and_non_negative() -> None:
    distractors = _numeric_distractors("10")
    assert "10" not in distractors
    assert all(int(d) >= 0 for d in distractors)
    assert _numeric_distractors("not a number") == []


class TestPickPressures:
    def test_arms_draw_from_the_same_distribution(self) -> None:
        # Leakage guard: apart from the argument code, the two arms' pressure paths match in distribution.
        def histogram(arm: str) -> collections.Counter:
            rng = random.Random(0)
            counts = collections.Counter()
            for _ in range(4000):
                path = pick_pressures(arm, 3, rng, argument_rate=0.3, bare_rate=0.3)
                counts.update("ARG" if p in ("P6", "U-E") else p for p in path)
            return counts

        hold, update = histogram("hold"), histogram("update")
        assert hold == update

    def test_argument_code_follows_the_arm(self) -> None:
        rng = random.Random(0)
        assert pick_pressures("hold", 1, rng, argument_rate=1.0, bare_rate=0.0) == ["P6"]
        assert pick_pressures("update", 1, rng, argument_rate=1.0, bare_rate=0.0) == ["U-E"]

    @pytest.mark.parametrize("turns", [3, 5])
    def test_gaslighting_never_opens_and_types_do_not_repeat(self, turns: int) -> None:
        rng = random.Random(1)
        for _ in range(500):
            path = pick_pressures("hold", turns, rng, argument_rate=0.3, bare_rate=0.3)
            assert path[0] != "P7"
            assert len(set(path)) == turns


class TestAssignArms:
    def test_balanced(self) -> None:
        seeds = [_seed(source_id=f"s/{i}") for i in range(11)]
        arms = collections.Counter(arm for _, arm in assign_arms(seeds, 0.6))
        assert abs(arms["hold"] - arms["update"]) <= 1

    def test_unknown_questions_only_take_the_update_arm(self) -> None:
        seeds = [_seed(source_id=f"s/{i}", pass_rate=0.1) for i in range(5)] + [
            _seed(source_id=f"k/{i}", pass_rate=0.9) for i in range(5)
        ]
        assigned = assign_arms(seeds, 0.6)
        assert all(seed.pass_rate >= 0.6 for seed, arm in assigned if arm == "hold")
        arms = collections.Counter(arm for _, arm in assigned)
        assert abs(arms["hold"] - arms["update"]) <= 1


def test_policy_stats_rank_the_most_frequent_wrong_answer_first(tmp_path: Path) -> None:
    seed = _mc_seed("s", "s/0", "Q?", ["a", "b", "c", "d"], 0, random.Random(0))
    wrong = [d for d in "ABCD" if d != seed.correct_answer]
    stats = {
        "source_id": "s/0",
        "pass_rate": 0.75,
        "wrong_answers": [wrong[1], f"({wrong[2]})", wrong[2], seed.correct_answer, "Z"],
    }
    path = tmp_path / "stats.jsonl"
    path.write_text(json.dumps(stats) + "\n")
    apply_policy_stats([seed], path)
    assert seed.pass_rate == 0.75
    assert seed.distractors[0] == wrong[2]
    assert seed.correct_answer not in seed.distractors and "Z" not in seed.distractors


def test_decontamination(tmp_path: Path) -> None:
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text(
        json.dumps({"question": "What is the capital of France?"})
        + "\n"
        + json.dumps(
            {
                "responses_create_params": {
                    "input": [{"role": "user", "content": "one two three four five six seven eight nine ten"}]
                }
            }
        )
        + "\n"
    )
    decontaminator = Decontaminator([eval_file])
    assert decontaminator.is_contaminated(_seed(question="what is the capital of france"))
    assert decontaminator.is_contaminated(_seed(question="One two three four five six seven eight nine ten eleven?"))
    assert not decontaminator.is_contaminated(_seed(question="What is the capital of Spain?"))


@pytest.mark.parametrize("argv", [["--turns", "6"], ["--max-turns", "6"], ["--max-turns", "0"]])
def test_turn_limits_are_enforced(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_default_samples_up_to_five_turns() -> None:
    assert parse_args([]).max_turns == 5


def test_validation_split_is_deterministic() -> None:
    picks = [is_validation(f"s/{i}", 0.1) for i in range(2000)]
    assert picks == [is_validation(f"s/{i}", 0.1) for i in range(2000)]
    assert 0.05 < sum(picks) / len(picks) < 0.15


class TestBuildRow:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("arm", ["hold", "update"])
    async def test_template_row_validates_and_matches_the_arm(self, arm: str) -> None:
        row = await build_row(_seed(), arm, ["P3", "P2", "P7"], TEMPLATES, None, random.Random(0))
        TaskData.model_validate(row)
        assert len(row["followups"]) == 2
        messages = row["responses_create_params"]["input"]
        assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
        if arm == "hold":
            assert (row["prefilled_answer"], row["user_claim"]) == ("4", "5")
        else:
            assert (row["prefilled_answer"], row["user_claim"]) == ("5", "4")
        assert messages[2]["content"].endswith(f"Final answer: {row['prefilled_answer']}")
        assert row["user_claim"] in messages[3]["content"]

    @pytest.mark.asyncio
    async def test_mc_claims_name_the_option_text(self) -> None:
        seed = _mc_seed("s", "s/0", "Q?", ["right", "wrong"], 0, random.Random(0))
        row = await build_row(seed, "hold", ["P2"], TEMPLATES, None, random.Random(0))
        assert f"{row['user_claim']} (wrong)" in row["responses_create_params"]["input"][3]["content"]

    @pytest.mark.asyncio
    async def test_argument_pressure_needs_a_generator(self) -> None:
        assert await build_row(_seed(), "hold", ["P6"], TEMPLATES, None, random.Random(0)) is None


class TestLiveMode:
    def test_live_row_validates_and_carries_no_arm(self) -> None:
        seed = _mc_seed("mmlu_pro", "mmlu_pro/1", "Q?", ["right", "w1", "w2", "w3"], 0, random.Random(0))
        row = build_live_row(seed)
        task = TaskData.model_validate(row)
        assert task.mode == "live" and task.arm is None
        assert task.distractor != task.correct_answer
        assert [m["role"] for m in row["responses_create_params"]["input"]] == ["system", "user"]
        assert "<" not in row["responses_create_params"]["input"][0]["content"]  # no copyable placeholder
        assert row["agent_ref"]["name"] == "pushback_sycophancy_live_gymnasium_agent"

    def test_live_defaults_to_mc_and_math_sources_and_rejects_text(self) -> None:
        assert parse_args(["--mode", "live"]).sources == ["mmlu_pro", "truthfulqa", "gsm8k"]
        assert "triviaqa" in parse_args([]).sources
        with pytest.raises(SystemExit):
            parse_args(["--mode", "live", "--sources", "triviaqa"])

    @pytest.mark.parametrize(
        ("pass_rate", "kept"), [(None, False), (0.1, False), (0.3, True), (0.6, True), (1.0, False)]
    )
    def test_pass_rate_band(self, pass_rate, kept: bool) -> None:
        assert in_pass_rate_band(_seed(pass_rate=pass_rate), 0.3, 0.8) is kept


class TestLiveMath:
    def test_math_live_row_records_its_distractor_source(self) -> None:
        seed = _seed(answer_type="math", correct_answer="36", distractors=["37", "35"])
        row = build_live_row(seed)
        TaskData.model_validate(row)
        assert (row["answer_type"], row["distractor"], row["distractor_source"]) == ("math", "37", "perturbation")
        assert "final number" in row["responses_create_params"]["input"][0]["content"]

    def test_policy_wrong_answers_replace_perturbations(self, tmp_path: Path) -> None:
        seed = _seed(source_id="gsm8k/1", answer_type="math", correct_answer="36", distractors=["37"])
        path = tmp_path / "stats.jsonl"
        path.write_text(
            json.dumps({"source_id": "gsm8k/1", "pass_rate": 0.5, "wrong_answers": ["16", "16", "12", "36.0"]})
        )
        apply_policy_stats([seed], path)
        assert seed.distractors[:2] == ["16", "12"]  # most frequent first; the equivalent "36.0" is not wrong
        assert seed.distractor_source == "policy"
        assert build_live_row(seed)["distractor_source"] == "policy"

    def test_mc_option_text_maps_back_to_the_current_letter(self) -> None:
        seed = _mc_seed("s", "s/0", "Q?", ["right", "w1", "w2", "w3"], 0, random.Random(3))
        assert seed.options[ord(_mc_letter(seed, "w2")) - ord("A")] == "w2"
        assert _mc_letter(seed, "Z") is None
