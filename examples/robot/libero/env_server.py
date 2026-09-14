"""Serve headless LIBERO simulator sessions over HTTP."""

import argparse
import asyncio
import base64
import io
import os
import traceback
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from robosuite.utils import transform_utils as transform  # noqa: E402


CONTROL_HZ = 20


def camera_rgb(image):
    """Return one camera frame in the orientation expected by the agent."""
    return np.ascontiguousarray(np.asarray(image)[::-1, ::-1].astype(np.uint8))


def jpeg_data_url(rgb_image):
    """Encode one RGB frame for an OpenAI image message."""
    jpeg_buffer = io.BytesIO()
    Image.fromarray(rgb_image).save(jpeg_buffer, format="JPEG", quality=90)
    base64_jpeg = base64.b64encode(jpeg_buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + base64_jpeg


def action_vector(name, values, max_length):
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain 3 finite numbers")
    if np.linalg.norm(vector) > max_length:
        raise ValueError(f"{name} length must not exceed {max_length}")
    return vector


class LiberoSimulator:
    """Own one LIBERO environment and its episode video."""

    def __init__(self, *, suite, task_id, init_state_id, video_path=None):
        benchmark_dict = benchmark.get_benchmark_dict()
        if suite not in benchmark_dict:
            raise ValueError(f"Unknown LIBERO suite: {suite}")
        task_suite = benchmark_dict[suite]()
        if not 0 <= task_id < task_suite.n_tasks:
            raise ValueError(f"task_id out of range: {task_id}")
        initial_states = task_suite.get_task_init_states(task_id)
        if not 0 <= init_state_id < len(initial_states):
            raise ValueError(f"init_state_id out of range: {init_state_id}")

        task = task_suite.get_task(task_id)
        bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        environment = OffScreenRenderEnv(bddl_file_name=str(bddl_file), camera_heights=512, camera_widths=512, control_freq=CONTROL_HZ)
        try:
            environment.reset()
            observation = environment.set_init_state(initial_states[init_state_id])
            for _ in range(10):
                observation, _, _, _ = environment.step([0.0] * 6 + [-1.0])
        except BaseException:
            environment.close()
            raise

        self.environment = environment
        self.observation = observation
        self.task_description = task.language
        self.task_completed = environment.check_success()
        self.video_path = Path(video_path) if video_path else None
        self.video_frames = [camera_rgb(observation["agentview_image"])] if video_path else []

    def close(self):
        try:
            if self.video_path and self.video_frames:
                self.video_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(str(self.video_path), self.video_frames, fps=CONTROL_HZ)
        finally:
            self.environment.close()

    def observe(self):
        """Return only the task state needed by the agent."""
        gripper_qpos = self.observation["robot0_gripper_qpos"]
        return {
            "success": self.task_completed,
            "gripper_opening_m": round(float(gripper_qpos[0] - gripper_qpos[1]), 4),
            "agentview_image": jpeg_data_url(camera_rgb(self.observation["agentview_image"])),
            "wrist_image": jpeg_data_url(camera_rgb(self.observation["robot0_eye_in_hand_image"])),
        }

    def act(self, *, move_m, rotate_rad, gripper):
        if gripper not in {"open", "close"}:
            raise ValueError("gripper must be open or close")

        start_position = np.asarray(self.observation["robot0_eef_pos"]).copy()
        start_quaternion = np.asarray(self.observation["robot0_eef_quat"]).copy()
        move_m = action_vector("move_m", move_m, 0.3)
        rotate_rad = action_vector("rotate_rad", rotate_rad, 1.0)
        gripper_action = {"open": -1.0, "close": 1.0}[gripper]

        # Follow a one-second linear pose trajectory at 20 Hz.
        for step in range(1, CONTROL_HZ + 1):
            if self.task_completed:
                break
            fraction = step / CONTROL_HZ
            target_position = start_position + move_m * fraction
            target_quaternion = transform.quat_multiply(
                transform.axisangle2quat(rotate_rad * fraction), start_quaternion,
            )
            current_position = np.asarray(self.observation["robot0_eef_pos"])
            position_action = np.clip(
                (target_position - current_position) / 0.05, -1.0, 1.0
            )

            rotation_error = transform.quat_distance(
                target_quaternion, np.asarray(self.observation["robot0_eef_quat"]),
            )
            if rotation_error[3] < 0.0:
                rotation_error = -rotation_error
            rotation_action = np.clip(
                transform.quat2axisangle(rotation_error) / 0.5, -1.0, 1.0
            )

            (
                self.observation,
                _,
                _,
                _,
            ) = self.environment.step(
                np.concatenate(
                    (position_action, rotation_action, [gripper_action])
                )
            )
            self.task_completed = self.environment.check_success()
            if self.video_path:
                self.video_frames.append(
                    camera_rgb(self.observation["agentview_image"])
                )

    def inspect_task_state(self):
        base_environment = self.environment.env
        goal = base_environment.parsed_problem["goal_state"][0]
        target_position = base_environment.object_states_dict[goal[1]].get_geom_state()["pos"]
        gripper_position = np.asarray(self.observation["robot0_eef_pos"])
        target_xy = target_position[:2] - gripper_position[:2]

        target = base_environment.objects_dict.get(goal[1])
        if target is None:
            left_contact = None
            right_contact = None
            both_fingerpads_contact = None
        else:
            gripper = base_environment.robots[0].gripper
            left_contact = base_environment.check_contact(
                gripper.important_geoms["left_fingerpad"], target.contact_geoms
            )
            right_contact = base_environment.check_contact(
                gripper.important_geoms["right_fingerpad"], target.contact_geoms
            )
            both_fingerpads_contact = base_environment._check_grasp(gripper, target)

        goal_xy = None
        if len(goal) == 3:
            goal_position = base_environment.object_states_dict[goal[2]].get_geom_state()["pos"]
            goal_xy = goal_position[:2] - target_position[:2]

        return {
            "target_body_origin_xy_relative_to_eef_m": np.round(target_xy, 4).tolist(),
            "eef_height_above_target_body_origin_m": round(float(gripper_position[2] - target_position[2]), 4),
            "left_fingerpad_contacts_target": left_contact,
            "right_fingerpad_contacts_target": right_contact,
            "both_fingerpads_contact_target": both_fingerpads_contact,
            "goal_body_origin_xy_relative_to_target_body_origin_m": np.round(goal_xy, 4).tolist() if goal_xy is not None else None,
        }


sessions = {}
app = FastAPI()


def get_session(session_id):
    simulator = sessions.get(session_id)
    if simulator is None:
        raise ValueError(f"Unknown simulator session: {session_id}")
    return simulator


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/sessions/{session_id}/new")
async def new(session_id: str, params: dict):
    if session_id in sessions:
        raise ValueError(f"Simulator session already exists: {session_id}")
    await app.state.session_slots.acquire()
    try:
        video_path = None
        if app.state.video_dir is not None:
            video_path = app.state.video_dir / f"{params['suite']}_task{params['task_id']}_init{params['init_state_id']}_{session_id}.mp4"
        sessions[session_id] = LiberoSimulator(**params, video_path=video_path)
    except BaseException:
        app.state.session_slots.release()
        raise
    return sessions[session_id].task_description


@app.post("/sessions/{session_id}/observe")
async def observe(session_id: str):
    return get_session(session_id).observe()


@app.post("/sessions/{session_id}/act")
async def act(session_id: str, params: dict):
    return get_session(session_id).act(**params)


@app.post("/sessions/{session_id}/inspect_task_state")
async def inspect_task_state(session_id: str):
    return get_session(session_id).inspect_task_state()


@app.post("/sessions/{session_id}/close")
async def close(session_id: str):
    get_session(session_id).close()
    sessions.pop(session_id)
    app.state.session_slots.release()


@app.exception_handler(Exception)
async def handle_error(_, error: Exception):
    traceback.print_exception(type(error), error, error.__traceback__)
    return JSONResponse(status_code=500, content={"error": {"type": type(error).__name__, "message": str(error)}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-sessions", type=int, default=1)
    parser.add_argument("--video-dir", type=Path)
    args = parser.parse_args()
    if args.max_sessions < 1:
        parser.error("--max-sessions must be at least 1")
    app.state.session_slots = asyncio.Semaphore(args.max_sessions)
    app.state.video_dir = args.video_dir.resolve() if args.video_dir else None

    try:
        uvicorn.run(app, host="0.0.0.0", port=8010)
    finally:
        for simulator in sessions.values():
            simulator.close()


if __name__ == "__main__":
    main()
