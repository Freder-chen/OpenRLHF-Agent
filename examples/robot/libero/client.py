"""HTTP client and tools for one LIBERO simulator session."""

from __future__ import annotations

import json
import math
from typing import Any
from uuid import uuid4

import httpx

from openrlhf_agent.utils.types import Message, ToolCall


ACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "act",
        "description": "Move the end effector relative to its current pose, control the gripper, and return camera views and task status.",
        "parameters": {
            "type": "object",
            "properties": {
                "move_m": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "World-frame displacement [dx, dy, dz] in meters with a maximum length of 0.3 m. +x moves forward; -x moves backward. +y moves to the robot's left; -y moves to the robot's right. +z moves up; -z moves down.",
                    "items": {"type": "number", "minimum": -0.3, "maximum": 0.3},
                },
                "rotate_rad": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "World-frame rotation vector [rx, ry, rz] with a maximum angle of 1 radian. The vector direction is the axis, its length is the angle, and positive values follow the right-hand rule.",
                    "items": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                },
                "gripper": {
                    "type": "string",
                    "enum": ["open", "close"],
                    "description": "Command applied to the gripper throughout the motion.",
                },
            },
            "required": ["move_m", "rotate_rad", "gripper"],
            "additionalProperties": False,
        },
    },
}

PRIVILEGED_FEEDBACK_TOOL = {
    "type": "function",
    "function": {
        "name": "privileged_feedback",
        "description": "Training only. Calling this tool incurs a penalty, so use it only when necessary. Returns the target position, gripper contact, and goal offset.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}


def validate_action(arguments: dict) -> None:
    if set(arguments) != {"move_m", "rotate_rad", "gripper"}:
        raise ValueError("act requires exactly move_m, rotate_rad, and gripper")
    if arguments["gripper"] not in ("open", "close"):
        raise ValueError("gripper must be open or close")

    validate_vector("move_m", arguments["move_m"], 0.3)
    validate_vector("rotate_rad", arguments["rotate_rad"], 1.0)


def validate_vector(name: str, values: object, max_length: float) -> None:
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"{name} must contain 3 numbers")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain 3 finite numbers")
    if math.hypot(*values) > max_length:
        raise ValueError(f"{name} length must not exceed {max_length}")


class LiberoClient:
    """Manage one LIBERO episode through a local HTTP simulator server."""

    def __init__(
        self,
        *,
        suite: str,
        task_id: int,
        init_state_id: int,
        base_url: str = "http://127.0.0.1:8010",
        privileged_feedback: bool = False,
    ) -> None:
        self.session_id = uuid4().hex
        self.config = {
            "suite": suite,
            "task_id": task_id,
            "init_state_id": init_state_id,
        }
        self.http = httpx.AsyncClient(base_url=base_url, timeout=None)
        self.task_description: str | None = None
        self.observation: dict[str, Any] | None = None
        self.tools = [ACTION_TOOL]
        self.privileged_feedback = privileged_feedback
        if privileged_feedback:
            self.tools.append(PRIVILEGED_FEEDBACK_TOOL)

    @property
    def success(self) -> bool:
        return bool(self.observation and self.observation["success"])

    @staticmethod
    def format_observation(
        observation: dict[str, Any],
        *,
        task_description: str | None = None,
    ) -> list[dict[str, Any]]:
        content = [
            {
                "type": "text",
                "text": f"Task completed: {'yes' if observation['success'] else 'no'}",
            },
            {
                "type": "text",
                "text": f'\nGripper opening: {observation["gripper_opening_m"]} m',
            },
            {"type": "text", "text": "\nAgent view:"},
            {
                "type": "image_url",
                "image_url": {"url": observation["agentview_image"]},
            },
            {"type": "text", "text": "\nWrist view:"},
            {
                "type": "image_url",
                "image_url": {"url": observation["wrist_image"]},
            },
        ]
        if task_description is not None:
            content.insert(0, {"type": "text", "text": f"Task: {task_description}\n"})
        return content

    async def new(self) -> list[Message]:
        self.task_description = await self._request("new", self.config)
        self.observation = await self._request("observe", {})
        return [
            Message(
                role="user",
                content=self.format_observation(
                    self.observation,
                    task_description=self.task_description,
                ),
            )
        ]

    async def execute_tool(
        self,
        call: ToolCall,
    ) -> str | list[dict[str, Any]]:
        arguments = call.arguments or {}
        if call.name == "act":
            validate_action(arguments)
            await self._request("act", arguments)
            self.observation = await self._request("observe", {})
            return self.format_observation(self.observation)

        if call.name == "privileged_feedback" and self.privileged_feedback:
            if arguments:
                raise ValueError("privileged_feedback takes no arguments")
            return json.dumps(
                await self._request("inspect_task_state", {}),
                ensure_ascii=False,
            )

        raise KeyError(f"Unknown tool '{call.name}'.")

    async def close(self) -> None:
        if self.http.is_closed:
            return
        try:
            if self.task_description is not None:
                await self._request("close", {})
        finally:
            self.task_description = None
            self.observation = None
            await self.http.aclose()

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
    ) -> Any:
        response = await self.http.post(f"/sessions/{self.session_id}/{method}", json=params)
        payload = response.json()
        if response.is_error:
            raise RuntimeError(str(payload["error"].get("message", "Unknown simulator error")))
        return payload
