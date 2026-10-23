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
"""Check that a row's arm cannot be predicted from its pushback text alone.

If a classifier that sees only the user's pushbacks predicts hold vs. update
well above chance, the policy can learn "this wording means switch" instead of
checking the claim. Target: cross-validated accuracy below 0.6.

    uv pip install scikit-learn
    python resources_servers/pushback_sycophancy/scripts/leakage_check.py resources_servers/pushback_sycophancy/data/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def pushback_text(row: dict) -> str:
    pushbacks = [row["responses_create_params"]["input"][-1]["content"], *row.get("followups", [])]
    text = "\n".join(pushbacks)
    # Mask the claim itself: it is question-specific, not a style cue, and an MC
    # letter would otherwise let the classifier memorise per-question answers.
    return text.replace(row["user_claim"], "<claim>") if row.get("user_claim") else text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args(argv)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    texts = [re.sub(r"\s+", " ", pushback_text(row)) for row in rows]
    arms = [row["arm"] for row in rows]
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=1000))
    accuracy = cross_val_score(model, texts, arms, cv=5, scoring="accuracy").mean()
    majority = max(arms.count("hold"), arms.count("update")) / len(arms)
    print(f"rows={len(rows)} arm-prediction accuracy={accuracy:.3f} (majority baseline {majority:.3f})")
    if accuracy >= args.threshold:
        print(f"LEAK: accuracy >= {args.threshold}; the pushback wording reveals the arm", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
