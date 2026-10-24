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
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.gymnasium import EnvResetRequest, EnvStepRequest
from resources_servers.pushback_sycophancy.app import PushbackSycophancyConfig, PushbackSycophancyServer


def _load(name: str) -> list[dict]:
    path = Path(__file__).parents[1] / "data" / name
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


EXAMPLE_ROWS = _load("example.jsonl")
EXAMPLE_LIVE_ROWS = _load("example_live.jsonl")


def make_server(*, judge: bool = False, **overrides) -> PushbackSycophancyServer:
    config = PushbackSycophancyConfig(
        host="",
        port=0,
        entrypoint="",
        name="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge") if judge else None,
        **overrides,
    )
    return PushbackSycophancyServer(config=config, server_client=MagicMock(spec=ServerClient))


def make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0.0,
        model="model",
        object="response",
        output=[
            {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def http_request(session_id: str = "sid") -> SimpleNamespace:
    return SimpleNamespace(session={SESSION_ID_KEY: session_id})


async def run_episode(server: PushbackSycophancyServer, row: dict, replies, session_id: str = "sid"):
    """Drive one episode through the HTTP endpoints, as gymnasium_agent does.

    replies is a callable (row, turn_index) -> str producing each policy reply.
    Returns the terminal EnvStepResponse and the observations returned along the way.
    """
    request = http_request(session_id)
    await server._reset_endpoint(EnvResetRequest(**row), request)
    observations = []
    # Prefill rows take 1 + len(followups) turns; live rows take 2 (or 1 with no first answer).
    for turn in range(len(row.get("followups", [])) + 2):
        step = await server._step_endpoint(EnvStepRequest(**row, response=make_response(replies(row, turn))), request)
        if step.terminated or step.truncated:
            return step, observations
        observations.append(step.observation)
    raise AssertionError("episode did not terminate after the last pushback")


@pytest.fixture
def example_rows() -> list[dict]:
    return [json.loads(json.dumps(row)) for row in EXAMPLE_ROWS]


@pytest.fixture
def example_live_rows() -> list[dict]:
    return [json.loads(json.dumps(row)) for row in EXAMPLE_LIVE_ROWS]
