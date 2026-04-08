"""
OpenEnv SDK Client for the Enterprise Task Dependency Scheduler.
================================================================
Subclasses openenv_core.HTTPEnvClient so the environment can be
used with `from_docker_image()` for the hackathon validator.

Usage:
    from scheduler_client import SchedulerEnvClient

    # From a running Docker container
    env = SchedulerEnvClient.from_docker_image("enterprise-task-scheduler")
    result = env.reset()
    result = env.step({"action_type": "assign", "task_id": "start", "worker_id": 0})
    env.close()

    # From a running server URL
    env = SchedulerEnvClient("http://localhost:7860")
    result = env.reset()
"""

from __future__ import annotations

from typing import Any, Optional

from openenv_core.http_env_client import HTTPEnvClient
from openenv_core.client_types import StepResult


class SchedulerObservation:
    """Lightweight observation wrapper for the client side."""

    def __init__(self, data: dict):
        self._data = data
        self.current_time: int = data.get("current_time", 0)
        self.ready_tasks: list = data.get("ready_tasks", [])
        self.idle_workers: list = data.get("idle_workers", [])
        self.workers: list = data.get("workers", [])
        self.running_tasks: dict = data.get("running_tasks", {})
        self.completed_tasks: list = data.get("completed_tasks", [])
        self.pending_tasks: list = data.get("pending_tasks", [])
        self.all_tasks: list = data.get("all_tasks", [])
        self.num_workers: int = data.get("num_workers", 0)
        self.is_done: bool = data.get("is_done", False)
        self.last_action_valid: bool = data.get("last_action_valid", True)
        self.message: str = data.get("message", "")

    def to_dict(self) -> dict:
        return self._data


class SchedulerEnvClient(HTTPEnvClient):
    """
    OpenEnv-compatible client for the Enterprise Task Scheduler.

    Wraps the HTTP API so the hackathon validator can spin up
    the Docker container and interact with it using the standard
    openenv-core lifecycle.
    """

    def _step_payload(self, action: dict) -> dict:
        """
        Convert action dict to the JSON body for /step.

        The SDK wraps this into {"action": <payload>, "timeout_s": N}
        automatically. We just return the raw action fields.
        """
        if isinstance(action, dict):
            return action
        # Fallback: if someone passes an object with __dict__
        return vars(action)

    def _parse_result(self, payload: dict) -> StepResult:
        """
        Parse the JSON response from /reset or /step into a StepResult.

        /reset returns a bare observation dict.
        /step returns {"observation": {...}, "reward": float, "done": bool}.
        """
        # /step response format
        if "observation" in payload:
            obs = SchedulerObservation(payload["observation"])
            return StepResult(
                observation=obs,
                reward=payload.get("reward", 0.0),
                done=payload.get("done", False),
            )

        # /reset response format — bare observation
        obs = SchedulerObservation(payload)
        return StepResult(
            observation=obs,
            reward=0.0,
            done=False,
        )

    def _parse_state(self, payload: dict) -> dict:
        """Parse the JSON response from /state."""
        return payload

    def reset_with_task(self, task_name: str = "easy", seed: Optional[int] = None) -> StepResult:
        """
        Reset with a specific task name and seed.

        The base SDK reset() sends {}, which defaults to 'easy'.
        This method allows explicitly choosing the task.
        """
        import requests

        body = {"task_name": task_name}
        if seed is not None:
            body["seed"] = seed

        r = self._http.post(
            f"{self._base}/reset",
            json=body,
            headers=self._headers,
            timeout=self._timeout,
        )
        r.raise_for_status()
        return self._parse_result(r.json())

    def get_grader(self) -> dict:
        """Get grading info from /grader endpoint."""
        r = self._http.get(
            f"{self._base}/grader",
            headers=self._headers,
            timeout=self._timeout,
        )
        r.raise_for_status()
        return r.json()
