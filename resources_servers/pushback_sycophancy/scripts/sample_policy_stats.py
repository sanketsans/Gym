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
"""Sample the base policy on live-mode rows and record its pass rate and wrong answers.

The output feeds ``build_seed.py --mode live --policy-stats``, which keeps questions in
a mid pass-rate band and uses the policy's own most frequent wrong answer as the
distractor the user pushes on the hold arm. Each reply is graded with the
environment's own rule (``grade_answer``), so "wrong" means the same thing here as in
training.

    # 1. Rows without stats (math distractors are placeholders at this point).
    python resources_servers/pushback_sycophancy/scripts/build_seed.py --mode live --out-dir /tmp/seed
    # 2. Sample the base policy, 8 replies per question.
    python resources_servers/pushback_sycophancy/scripts/sample_policy_stats.py \\
        --input /tmp/seed/train_live.jsonl --base-url http://localhost:8000/v1 --model my-policy \\
        --output /tmp/seed/policy_stats.jsonl
    # 3. Rebuild with the stats.
    python resources_servers/pushback_sycophancy/scripts/build_seed.py --mode live \\
        --policy-stats /tmp/seed/policy_stats.jsonl

Output, one line per question: ``{"source_id", "pass_rate", "num_samples",
"wrong_answers"}``. MC wrong answers are stored as option text, so they map back to
the right letter even if a later build shuffles the options differently.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional


SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR.parents[1]))

from resources_servers.pushback_sycophancy.answer_extraction import (  # noqa: E402
    LETTERS,
    extract_final_answer,
    grade_answer,
    math_names_other_values,
    mc_names_other_options,
    normalize_mc,
)


def summarize(row: dict, replies: list[str]) -> dict:
    """Pass rate and wrong answers for one question, graded as the environment grades turn 0."""
    options = row.get("options")
    wrong: list[str] = []
    passes = 0
    for reply in replies:
        answer = extract_final_answer(reply)
        if grade_answer(answer, row["correct_answer"], row["answer_type"], options=options):
            passes += 1
            continue
        if answer is None or math_names_other_values(answer) or mc_names_other_options(answer, options):
            continue  # not a committed answer: counts against the pass rate but is not a distractor
        if row["answer_type"] == "mc":
            letter = normalize_mc(answer)
            if not options or letter is None or letter not in LETTERS[: len(options)]:
                continue
            answer = options[LETTERS.index(letter)]
        wrong.append(answer)
    return {
        "source_id": row["source_id"],
        "pass_rate": passes / len(replies) if replies else 0.0,
        "num_samples": len(replies),
        "wrong_answers": wrong,
    }


async def sample(session, args: argparse.Namespace, row: dict, semaphore: asyncio.Semaphore) -> Optional[dict]:
    payload = {
        "model": args.model,
        "messages": row["responses_create_params"]["input"],
        "n": args.samples,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    headers = {"Authorization": f"Bearer {os.environ[args.api_key_env]}"} if os.environ.get(args.api_key_env) else {}
    async with semaphore:
        for attempt in range(3):
            try:
                async with session.post(
                    args.base_url.rstrip("/") + "/chat/completions", json=payload, headers=headers
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                replies = [(c["message"].get("content") or "") for c in data["choices"]]
                return summarize(row, replies)
            except Exception as e:  # noqa: BLE001 - offline script; a failed row is reported and skipped
                if attempt == 2:
                    print(f"{row['source_id']}: sampling failed: {type(e).__name__}: {e}", file=sys.stderr)
                await asyncio.sleep(2**attempt)
    return None


async def main(args: argparse.Namespace) -> None:
    import aiohttp

    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("mode") == "live"][: args.limit or None]
    semaphore = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.timeout)) as session:
        stats = await asyncio.gather(*(sample(session, args, row, semaphore) for row in rows))
    stats = [s for s in stats if s is not None]
    with open(args.output, "w") as f:
        for s in stats:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with_wrong = sum(1 for s in stats if s["wrong_answers"])
    mean = sum(s["pass_rate"] for s in stats) / len(stats) if stats else 0.0
    print(
        f"wrote {len(stats)}/{len(rows)} stats to {args.output}: mean pass rate {mean:.2f}, {with_wrong} with a wrong answer"
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Live-mode JSONL from build_seed.py --mode live.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint serving the base policy.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="POLICY_API_KEY")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--limit", type=int, default=0, help="Sample only the first N rows (0: all).")
    return parser.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
