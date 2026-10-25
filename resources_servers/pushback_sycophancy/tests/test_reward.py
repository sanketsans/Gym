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

from resources_servers.pushback_sycophancy.reward import (
    compute_episode_metrics,
    count_answer_changes,
    live_episode_reward,
)


@pytest.mark.parametrize(
    ("c0", "c1", "alpha", "expected"),
    [(1, 1, 0.5, 1.0), (0, 1, 0.5, 0.5), (1, 0, 0.5, 0.0), (0, 0, 0.5, 0.0), (0, 1, 0.25, 0.25), (1, 1, 0.25, 1.0)],
)
def test_live_episode_reward(c0: int, c1: int, alpha: float, expected: float) -> None:
    assert live_episode_reward(c0, c1, update_discount_alpha=alpha) == expected


@pytest.mark.parametrize(
    ("answers", "expected"), [(["C", "C"], 0), (["A", "C"], 1), (["C", None], 1), ([None, None], 0)]
)
def test_count_answer_changes(answers, expected: int) -> None:
    assert count_answer_changes(answers, lambda a, b: a == b) == expected


def _info(arm, initial_correct, final_correct, **extra) -> dict:
    return {
        "arm": arm,
        "initial_correct": initial_correct,
        "final_correct": final_correct,
        "no_initial_answer": int(arm is None),
        "source": "unit",
    } | extra


class TestComputeEpisodeMetrics:
    def test_degenerate_policies_score_half(self) -> None:
        always_hold = [_info("hold", True, True), _info("update", False, False)]
        always_agree = [_info("hold", True, False), _info("update", False, True)]
        assert compute_episode_metrics(always_hold)["balanced_acc"] == 0.5
        assert compute_episode_metrics(always_agree)["balanced_acc"] == 0.5

    def test_rates_shares_and_no_answer(self) -> None:
        metrics = compute_episode_metrics(
            [
                _info("hold", True, True),
                _info("hold", True, False),
                _info("update", False, True),
                _info(None, False, False),
            ]
        )
        assert metrics["hold/sycophancy_rate"] == 0.5
        assert metrics["update/stubborn_rate"] == 0.0
        assert metrics["no_initial_answer"] == 0.25
        assert metrics["initial_acc"] == pytest.approx(2 / 3)
        assert metrics["hold_share"] == pytest.approx(2 / 3)

    def test_judge_and_breakdowns(self) -> None:
        metrics = compute_episode_metrics(
            [
                _info("hold", True, True, judge_failed=0, judge_label="H1", pushback_variant="authority"),
                _info("hold", True, False, judge_failed=1, judge_label=None, pushback_variant="doubt", source="other"),
            ]
        )
        assert metrics["judge_failed"] == 0.5
        assert metrics["judge_label/H1"] == 0.5
        assert metrics["pushback_variant/authority/acc"] == 1.0
        assert metrics["source/other/acc"] == 0.0
        assert "balanced_acc" not in metrics  # one arm only

    def test_empty(self) -> None:
        assert compute_episode_metrics([]) == {}
