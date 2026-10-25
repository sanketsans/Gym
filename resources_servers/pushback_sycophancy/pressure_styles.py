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
"""Pressure styles for user pushbacks, shared by the server and offline data builders.

The catalog (``prompts/pressure_styles.yaml``) gives each style an instruction and a few example
phrasings. The server uses it when a simulated user writes the pushback live; data builders use the
same file to write frozen pushbacks, so both describe a style identically. Dependency-light on purpose
(stdlib + PyYAML) so builders can import it without the server's requirements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import yaml


DEFAULT_PRESSURE_STYLES_PATH = Path(__file__).parent / "prompts" / "pressure_styles.yaml"


def load_pressure_styles(path: str | Path = DEFAULT_PRESSURE_STYLES_PATH) -> dict[str, dict]:
    """``{style: {"instruction", "content", "examples"}}`` from the catalog, validated.

    ``content`` is ``working`` (the pushback shows concrete working) or ``pressure`` (no facts,
    numbers, or working: the pressure is social only).
    """
    styles = (yaml.safe_load(Path(path).read_text()) or {}).get("styles") or {}
    if not styles:
        raise ValueError(f"no styles in pressure-style catalog {path}")
    for name, style in styles.items():
        if "{claim}" not in (style or {}).get("instruction", ""):
            raise ValueError(f"pressure style {name!r} in {path} needs an instruction mentioning {{claim}}")
        style.setdefault("examples", [])
        style.setdefault("content", "pressure")
        if style["content"] not in ("pressure", "working"):
            raise ValueError(f"pressure style {name!r} in {path}: content must be 'pressure' or 'working'")
    return styles


def tone_examples_line(examples: Sequence[str]) -> str:
    """A prompt line showing example phrasings for tone, or "" when there are none."""
    if not examples:
        return ""
    quoted = "; ".join(f'"{e}"' for e in examples)
    return f"\n- Tone examples from other users, for style only (do not copy them; X stands for the claim): {quoted}"
