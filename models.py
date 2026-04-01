"""
Pydantic Models for the Enterprise Task Dependency Scheduler
=============================================================
These define the "contract" between the agent and the environment.
OpenEnv uses Pydantic so that:
  - The API is self-documenting (agents can inspect the schema)
  - Invalid data is caught before it reaches the engine
  - Serialization over HTTP is handled automatically

There are 4 key models:
  1. SchedulerAction  — what the agent SENDS   (input)
  2. SchedulerObservation — what the agent SEES (output)
  3. SchedulerState  — episode metadata         (output)
  4. StepResult      — observation + reward + done (output)
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# 1. ACTION — What the agent sends each step
# ─────────────────────────────────────────────────────────────


class SchedulerAction(BaseModel):
    """
    The agent's decision at each step.

    Two possible actions:
      1. ASSIGN: Schedule a specific task on a specific worker.
         Requires task_id and worker_id.
      2. WAIT: Do nothing — advance simulation time to the next
         task completion event. Use when no ready tasks exist,
         or when you're strategically waiting.

    Example:
        {"action_type": "assign", "task_id": "d_compile", "worker_id": 0}
        {"action_type": "wait"}
    """
    action_type: Literal["assign", "wait"] = Field(
        description="Either 'assign' to schedule a task, or 'wait' to advance time."
    )
    task_id: Optional[str] = Field(
        default=None,
        description="ID of the task to schedule. Required when action_type='assign'."
    )
    worker_id: Optional[int] = Field(
        default=None,
        description="ID of the worker to assign the task to. Required when action_type='assign'."
    )


# ─────────────────────────────────────────────────────────────
# 2. OBSERVATION — What the agent sees
# ─────────────────────────────────────────────────────────────


class TaskInfo(BaseModel):
    """Info about a single task, as seen by the agent."""
    task_id: str
    duration: int
    dependencies: list[str]


class WorkerInfo(BaseModel):
    """Info about a single worker, as seen by the agent."""
    worker_id: int
    is_idle: bool
    current_task: Optional[str] = None
    finish_time: Optional[int] = None


class SchedulerObservation(BaseModel):
    """
    Everything the agent can see about the current state.

    This is returned after reset() and after each step().
    The agent uses this to decide its next action.

    Key fields:
      - ready_tasks: Tasks that CAN be scheduled right now
      - idle_workers: Workers that are free
      - running_tasks: Tasks currently executing with finish times
      - completed_tasks: Tasks already done
      - pending_tasks: Tasks not yet started (may have unmet deps)
    """
    current_time: int = Field(
        description="Current simulation time step."
    )
    ready_tasks: list[TaskInfo] = Field(
        description="Tasks whose dependencies are met and can be scheduled NOW."
    )
    idle_workers: list[int] = Field(
        description="Worker IDs that are currently free."
    )
    workers: list[WorkerInfo] = Field(
        description="Full state of all workers."
    )
    running_tasks: dict[str, int] = Field(
        description="Mapping of task_id → finish_time for currently running tasks."
    )
    completed_tasks: list[str] = Field(
        description="List of task IDs that have finished."
    )
    pending_tasks: list[str] = Field(
        description="Task IDs not yet started (may have unmet dependencies)."
    )
    all_tasks: list[TaskInfo] = Field(
        description="Complete list of all tasks in this episode (for reference)."
    )
    num_workers: int = Field(
        description="Total number of workers available."
    )
    is_done: bool = Field(
        description="True if all tasks are completed (episode over)."
    )
    last_action_valid: bool = Field(
        default=True,
        description="Whether the last action was valid. False if assignment failed."
    )
    message: str = Field(
        default="",
        description="Human-readable status message or error description."
    )


# ─────────────────────────────────────────────────────────────
# 3. STATE — Episode metadata
# ─────────────────────────────────────────────────────────────


class SchedulerState(BaseModel):
    """
    Episode-level metadata. Returned by state().
    This is NOT the game state — it's meta-information about the episode.
    """
    episode_id: str = Field(
        description="Unique identifier for this episode."
    )
    task_name: str = Field(
        description="Name of the current task/difficulty: 'easy', 'medium', or 'hard'."
    )
    step_count: int = Field(
        description="Number of step() calls made so far in this episode."
    )
    is_done: bool = Field(
        description="Whether the episode has ended."
    )
    current_score: Optional[float] = Field(
        default=None,
        description="Current grader score (only set when episode is done)."
    )
    optimal_makespan: Optional[int] = Field(
        default=None,
        description="Best possible makespan (only revealed when episode is done)."
    )
    agent_makespan: Optional[int] = Field(
        default=None,
        description="Agent's achieved makespan (only set when episode is done)."
    )
    seed: int = Field(
        description="Random seed used for this episode."
    )


# ─────────────────────────────────────────────────────────────
# 4. STEP RESULT — What step() returns
# ─────────────────────────────────────────────────────────────


class StepResult(BaseModel):
    """
    The complete return value of step().
    Combines observation, reward, and termination flag.
    """
    observation: SchedulerObservation
    reward: float = Field(
        description="Reward for this step. 0.0 during episode, final score at termination."
    )
    done: bool = Field(
        description="True if the episode is over (all tasks completed)."
    )


# ─────────────────────────────────────────────────────────────
# 5. TASK DESCRIPTION — For the /tasks endpoint
# ─────────────────────────────────────────────────────────────


class TaskDescription(BaseModel):
    """Describes one of the difficulty levels for the /tasks endpoint."""
    name: str
    difficulty: str
    num_tasks: int
    num_workers: int
    description: str


class TasksResponse(BaseModel):
    """Response for the /tasks endpoint."""
    environment_name: str
    tasks: list[TaskDescription]
    action_schema: dict
    observation_schema: dict


# ─────────────────────────────────────────────────────────────
# 6. GRADER & BASELINE RESPONSES
# ─────────────────────────────────────────────────────────────


class GraderResponse(BaseModel):
    """Response for the /grader endpoint."""
    episode_id: str
    task_name: str
    score: float = Field(ge=0.0, le=1.0)
    agent_makespan: int
    optimal_makespan: int
    is_done: bool


class BaselineResult(BaseModel):
    """Result for one task in the baseline run."""
    task_name: str
    score: float
    agent_makespan: int
    optimal_makespan: int
    agent_strategy: str


class BaselineResponse(BaseModel):
    """Response for the /baseline endpoint."""
    results: list[BaselineResult]
    average_score: float


# ─────────────────────────────────────────────────────────────
# 7. REQUEST MODELS — What the server receives
# ─────────────────────────────────────────────────────────────


class ResetRequest(BaseModel):
    """Request body for POST /reset."""
    task_name: str = Field(
        default="easy",
        description="Difficulty level: 'easy', 'medium', or 'hard'."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for episode generation. None = random."
    )


class BaselineRequest(BaseModel):
    """Request body for POST /baseline."""
    task_name: str = Field(
        default="easy",
        description="Difficulty level to run baselines on."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for the baseline episode."
    )
    strategies: list[str] = Field(
        default=["alphabetical", "shortest_first", "longest_first", "random", "critical_path"],
        description="List of baseline strategies to run."
    )
