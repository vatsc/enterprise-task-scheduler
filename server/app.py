"""
Enterprise Task Dependency Scheduler — FastAPI Server
======================================================
Exposes the OpenEnv API (reset/step/state) plus custom endpoints
(grader/baseline/tasks) over HTTP.

Run locally:
    uvicorn server.app:app --reload --port 8000

Or from project root:
    python -m uvicorn server.app:app --reload --port 8000
"""

from __future__ import annotations

import sys
import os

# Add parent directory to path so we can import our modules
# when running as `uvicorn server.app:app`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from environment import SchedulerEnvironment
from models import (
    BaselineRequest,
    BaselineResponse,
    GraderResponse,
    ResetRequest,
    SchedulerAction,
    SchedulerObservation,
    SchedulerState,
    StepResult,
    TaskDescription,
    TasksResponse,
)
from scheduler_core import (
    create_easy_scenario,
    create_hard_scenario,
    create_medium_scenario,
)


# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Enterprise Task Dependency Scheduler",
    description=(
        "An OpenEnv environment where an AI agent schedules dependent tasks "
        "across multiple workers to minimize total execution time (makespan). "
        "The agent must intelligently prioritize tasks at contention points "
        "where more tasks are ready than workers are available."
    ),
    version="1.0.0",
)

# CORS — allow everything for local dev and agent access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single environment instance (in-memory, no persistence needed)
env = SchedulerEnvironment()


# ─────────────────────────────────────────────────────────────
# Standard OpenEnv Endpoints
# ─────────────────────────────────────────────────────────────


@app.post("/reset", response_model=SchedulerObservation)
async def reset(request: ResetRequest = ResetRequest()) -> SchedulerObservation:
    """
    Start a new episode.

    Accepts optional task_name ('easy', 'medium', 'hard') and seed.
    Returns the initial observation with all task/worker info.
    """
    try:
        observation = env.reset(
            task_name=request.task_name,
            seed=request.seed,
        )
        return observation
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
async def step(action: SchedulerAction) -> StepResult:
    """
    Execute one agent action.

    Action is either:
      - {"action_type": "assign", "task_id": "...", "worker_id": N}
      - {"action_type": "wait"}

    Returns observation, reward (0.0 during play, final score at end),
    and done flag.

    Invalid actions return normally with last_action_valid=False and
    an explanatory message — they do NOT crash or return 400.
    """
    try:
        result = env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=SchedulerState)
async def state() -> SchedulerState:
    """
    Get episode metadata.

    During the episode: score and makespans are hidden.
    After the episode: reveals final score, optimal makespan, agent makespan.
    """
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────
# Custom Endpoints (required by hackathon spec)
# ─────────────────────────────────────────────────────────────


@app.get("/grader", response_model=GraderResponse)
async def grader() -> GraderResponse:
    """
    Get grading information for the current episode.

    Returns score (0.0 to 1.0), agent makespan, and optimal makespan.
    Score is only meaningful after the episode is done.
    """
    try:
        return env.grade()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/baseline", response_model=BaselineResponse)
async def baseline(
    request: BaselineRequest = BaselineRequest(),
) -> BaselineResponse:
    """
    Run built-in agent strategies and return their scores.

    This does NOT affect the current episode — it runs
    separate simulations for each strategy.

    Available strategies: alphabetical, shortest_first, longest_first,
    random, critical_path.
    """
    try:
        return env.run_baseline(
            task_name=request.task_name,
            seed=request.seed,
            strategies=request.strategies,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks", response_model=TasksResponse)
async def tasks() -> TasksResponse:
    """
    List available tasks (difficulty levels) and their schemas.

    Returns the environment name, task descriptions, and the
    JSON schemas for actions and observations.
    """
    # Get scenario info for each difficulty
    task_descriptions = []
    scenario_info = {
        "easy": ("Easy", "Fork-join DAG with 3 branches and 2 workers. "
                 "One key contention point where the critical path task "
                 "must be prioritized."),
        "medium": ("Medium", "Two-stage contention with 3 competing branches "
                   "and 2 workers. Requires identifying the critical path "
                   "through the longest branch."),
        "hard": ("Hard", "Multi-phase CI/CD pipeline with 14 tasks and "
                 "3 workers. Multiple contention points across build, test, "
                 "package, and deploy phases."),
    }

    for task_name, (difficulty, description) in scenario_info.items():
        tasks_list, num_workers = {
            "easy": create_easy_scenario,
            "medium": create_medium_scenario,
            "hard": create_hard_scenario,
        }[task_name](seed=0)

        task_descriptions.append(TaskDescription(
            name=task_name,
            difficulty=difficulty,
            num_tasks=len(tasks_list),
            num_workers=num_workers,
            description=description,
        ))

    return TasksResponse(
        environment_name="enterprise-task-scheduler",
        tasks=task_descriptions,
        action_schema=SchedulerAction.model_json_schema(),
        observation_schema=SchedulerObservation.model_json_schema(),
    )


# ─────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Simple health check endpoint."""
    return {"status": "ok", "environment": "enterprise-task-scheduler"}
