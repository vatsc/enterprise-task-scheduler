"""
Enterprise Task Dependency Scheduler — Phase 2+3: OpenEnv Wrapper
==================================================================
This module wraps the Phase 1 core engine (scheduler_core.py) with the
OpenEnv Gymnasium-style API: reset() / step() / state().

It translates between:
  - The agent's Pydantic models (SchedulerAction, SchedulerObservation, etc.)
  - The core engine's dataclasses (Task, SimulationState, SchedulerSimulator)

This is the ONLY module the FastAPI server needs to import.
"""

from __future__ import annotations

import random
import uuid
from typing import Optional

from models import (
    BaselineResponse,
    BaselineResult,
    GraderResponse,
    SchedulerAction,
    SchedulerObservation,
    SchedulerState,
    StepResult,
    TaskInfo,
    WorkerInfo,
)
from scheduler_core import (
    SchedulerSimulator,
    Task,
    compute_optimal_makespan,
    create_easy_scenario,
    create_hard_scenario,
    create_medium_scenario,
    grade_episode,
    run_alphabetical_agent,
    run_longest_first_agent,
    run_random_agent,
    run_shortest_first_agent,
)


# ─────────────────────────────────────────────────────────────
# Scenario registry — maps task_name → create function
# ─────────────────────────────────────────────────────────────

SCENARIOS = {
    "easy": create_easy_scenario,
    "medium": create_medium_scenario,
    "hard": create_hard_scenario,
}

# Baseline strategy registry — maps name → runner function
BASELINE_STRATEGIES = {
    "alphabetical": run_alphabetical_agent,
    "shortest_first": run_shortest_first_agent,
    "longest_first": run_longest_first_agent,
    "random": run_random_agent,
    "critical_path": compute_optimal_makespan,
}


class SchedulerEnvironment:
    """
    OpenEnv-compatible environment wrapper.

    Lifecycle:
        1. env = SchedulerEnvironment()
        2. obs = env.reset(task_name="easy", seed=42)
        3. result = env.step(SchedulerAction(action_type="assign", task_id="start", worker_id=0))
        4. ... repeat step() until result.done == True
        5. grader_info = env.grade()
        6. obs = env.reset(...)  # start new episode
    """

    def __init__(self):
        # Episode state — all None until reset() is called
        self._sim: Optional[SchedulerSimulator] = None
        self._tasks_list: Optional[list[Task]] = None
        self._num_workers: int = 0
        self._task_name: str = ""
        self._seed: int = 0
        self._episode_id: str = ""
        self._step_count: int = 0
        self._is_done: bool = False
        self._optimal_makespan: int = 0
        self._last_action_valid: bool = True
        self._last_message: str = ""
        self._initialized: bool = False

    # ─────────────────────────────────────────────────────────
    # reset() → SchedulerObservation
    # ─────────────────────────────────────────────────────────

    def reset(
        self,
        task_name: str = "easy",
        seed: Optional[int] = None,
    ) -> SchedulerObservation:
        """
        Initialize a new episode.

        Args:
            task_name: Difficulty level — "easy", "medium", or "hard".
            seed: Random seed for reproducibility. None = random.

        Returns:
            SchedulerObservation: The initial observation.

        Raises:
            ValueError: If task_name is not in {"easy", "medium", "hard"}.
        """
        if task_name not in SCENARIOS:
            raise ValueError(
                f"Unknown task_name '{task_name}'. "
                f"Must be one of: {list(SCENARIOS.keys())}"
            )

        # Generate seed if not provided
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        self._task_name = task_name
        self._seed = seed

        # Create scenario and simulator
        self._tasks_list, self._num_workers = SCENARIOS[task_name](seed=seed)
        self._sim = SchedulerSimulator(self._tasks_list, self._num_workers)
        self._sim.reset()

        # Compute optimal makespan for grading
        self._optimal_makespan = compute_optimal_makespan(
            self._tasks_list, self._num_workers
        )

        # Episode metadata
        short_id = uuid.uuid4().hex[:8]
        self._episode_id = f"{task_name}-{seed}-{short_id}"
        self._step_count = 0
        self._is_done = False
        self._last_action_valid = True
        self._last_message = "Episode started. Schedule tasks to minimize makespan."
        self._initialized = True

        return self._build_observation()

    # ─────────────────────────────────────────────────────────
    # step(action) → StepResult
    # ─────────────────────────────────────────────────────────

    def step(self, action: SchedulerAction) -> StepResult:
        """
        Execute one agent action and return the result.

        Args:
            action: SchedulerAction with action_type="assign" or "wait".

        Returns:
            StepResult with observation, reward, and done flag.

        Raises:
            RuntimeError: If called before reset().
            RuntimeError: If episode is already done.
        """
        if not self._initialized or self._sim is None:
            raise RuntimeError(
                "Environment not initialized. Call reset() first."
            )

        if self._is_done:
            raise RuntimeError(
                "Episode is already done. Call reset() to start a new episode."
            )

        self._step_count += 1
        self._last_action_valid = True
        self._last_message = ""

        if action.action_type == "assign":
            self._handle_assign(action)
        elif action.action_type == "wait":
            self._handle_wait()
        else:
            # Should never happen due to Pydantic validation, but be safe
            self._last_action_valid = False
            self._last_message = (
                f"Unknown action_type '{action.action_type}'. "
                f"Use 'assign' or 'wait'."
            )

        # Check if episode is done
        if self._sim.state.is_done:
            self._is_done = True
            if self._sim.state.total_time == 0:
                # Edge case: set total_time if advance_time didn't
                self._sim.state.total_time = self._sim.state.current_time

        # Compute reward
        reward = 0.0
        if self._is_done:
            reward = grade_episode(
                self._sim.state.total_time, self._optimal_makespan
            )

        return StepResult(
            observation=self._build_observation(),
            reward=reward,
            done=self._is_done,
        )

    def _handle_assign(self, action: SchedulerAction) -> None:
        """Process an 'assign' action with full validation."""
        sim = self._sim
        assert sim is not None

        # Validate required fields
        if action.task_id is None:
            self._last_action_valid = False
            self._last_message = "action_type='assign' requires 'task_id'."
            return

        if action.worker_id is None:
            self._last_action_valid = False
            self._last_message = "action_type='assign' requires 'worker_id'."
            return

        # Validate task_id exists
        if action.task_id not in sim.tasks:
            self._last_action_valid = False
            self._last_message = (
                f"Task '{action.task_id}' does not exist. "
                f"Valid tasks: {sorted(sim.tasks.keys())}"
            )
            return

        # Validate worker_id is in range
        if action.worker_id < 0 or action.worker_id >= self._num_workers:
            self._last_action_valid = False
            self._last_message = (
                f"Worker {action.worker_id} does not exist. "
                f"Valid workers: 0..{self._num_workers - 1}"
            )
            return

        # Validate task is ready (not already running/completed)
        ready_tasks = sim.get_ready_tasks()
        if action.task_id not in ready_tasks:
            if action.task_id in sim.state.completed_tasks:
                self._last_message = (
                    f"Task '{action.task_id}' is already completed."
                )
            elif action.task_id in sim.state.running_tasks:
                self._last_message = (
                    f"Task '{action.task_id}' is already running "
                    f"(finishes at t={sim.state.running_tasks[action.task_id]})."
                )
            else:
                deps = sim.tasks[action.task_id].dependencies
                unmet = [d for d in deps if d not in sim.state.completed_tasks]
                self._last_message = (
                    f"Task '{action.task_id}' has unmet dependencies: {unmet}"
                )
            self._last_action_valid = False
            return

        # Validate worker is idle
        idle_workers = sim.get_idle_workers()
        if action.worker_id not in idle_workers:
            worker = sim.state.workers[action.worker_id]
            self._last_message = (
                f"Worker {action.worker_id} is busy with '{worker.current_task}' "
                f"(finishes at t={worker.finish_time})."
            )
            self._last_action_valid = False
            return

        # All validation passed — assign!
        success = sim.assign_task(action.task_id, action.worker_id)
        if success:
            self._last_message = (
                f"Assigned '{action.task_id}' to worker {action.worker_id}. "
                f"Will finish at t={sim.state.running_tasks[action.task_id]}."
            )
        else:
            # Should not reach here after validation, but be safe
            self._last_action_valid = False
            self._last_message = "Assignment failed unexpectedly."

    def _handle_wait(self) -> None:
        """Process a 'wait' action — advance time to next completion."""
        sim = self._sim
        assert sim is not None

        if not sim.state.running_tasks:
            # Nothing is running — wait is a no-op but valid
            # Check if there are ready tasks the agent could assign instead
            ready = sim.get_ready_tasks()
            if ready:
                self._last_message = (
                    "Wait processed, but there are ready tasks you could assign: "
                    f"{ready}. No time advanced (nothing running)."
                )
            else:
                # No running tasks and no ready tasks = deadlock or done
                self._last_message = (
                    "Wait processed. No tasks running and no tasks ready."
                )
            return

        old_time = sim.state.current_time
        new_time = sim.advance_time()
        self._last_message = (
            f"Advanced time from t={old_time} to t={new_time}."
        )

    # ─────────────────────────────────────────────────────────
    # state() → SchedulerState
    # ─────────────────────────────────────────────────────────

    def state(self) -> SchedulerState:
        """
        Return episode-level metadata.

        During the episode, score/makespan fields are hidden.
        After the episode ends, they are revealed.
        """
        if not self._initialized:
            raise RuntimeError(
                "Environment not initialized. Call reset() first."
            )

        if self._is_done:
            agent_makespan = self._sim.state.total_time if self._sim else 0
            return SchedulerState(
                episode_id=self._episode_id,
                task_name=self._task_name,
                step_count=self._step_count,
                is_done=True,
                current_score=grade_episode(agent_makespan, self._optimal_makespan),
                optimal_makespan=self._optimal_makespan,
                agent_makespan=agent_makespan,
                seed=self._seed,
            )

        return SchedulerState(
            episode_id=self._episode_id,
            task_name=self._task_name,
            step_count=self._step_count,
            is_done=False,
            current_score=None,
            optimal_makespan=None,
            agent_makespan=None,
            seed=self._seed,
        )

    # ─────────────────────────────────────────────────────────
    # grade() → GraderResponse
    # ─────────────────────────────────────────────────────────

    def grade(self) -> GraderResponse:
        """
        Return grading information for the current episode.

        Can be called at any time, but score is only meaningful
        after the episode is done.
        """
        if not self._initialized:
            raise RuntimeError(
                "Environment not initialized. Call reset() first."
            )

        agent_makespan = self._sim.state.total_time if self._sim and self._is_done else 0
        score = grade_episode(agent_makespan, self._optimal_makespan) if self._is_done else 0.0

        return GraderResponse(
            episode_id=self._episode_id,
            task_name=self._task_name,
            score=score,
            agent_makespan=agent_makespan if self._is_done else 0,
            optimal_makespan=self._optimal_makespan,
            is_done=self._is_done,
        )

    # ─────────────────────────────────────────────────────────
    # run_baseline() → BaselineResponse
    # ─────────────────────────────────────────────────────────

    def run_baseline(
        self,
        task_name: str = "easy",
        seed: Optional[int] = None,
        strategies: Optional[list[str]] = None,
    ) -> BaselineResponse:
        """
        Run built-in agent strategies and return their scores.

        This does NOT affect the current episode — it creates
        a separate simulation for each strategy.

        Args:
            task_name: Difficulty level.
            seed: Random seed. None = 42 (deterministic default).
            strategies: Which strategies to run. None = all.

        Returns:
            BaselineResponse with results for each strategy.
        """
        if task_name not in SCENARIOS:
            raise ValueError(
                f"Unknown task_name '{task_name}'. "
                f"Must be one of: {list(SCENARIOS.keys())}"
            )

        if seed is None:
            seed = 42

        if strategies is None:
            strategies = list(BASELINE_STRATEGIES.keys())

        tasks_list, num_workers = SCENARIOS[task_name](seed=seed)
        optimal = compute_optimal_makespan(tasks_list, num_workers)

        results: list[BaselineResult] = []
        for strategy_name in strategies:
            if strategy_name not in BASELINE_STRATEGIES:
                continue

            runner = BASELINE_STRATEGIES[strategy_name]

            # run_random_agent has a different signature (extra seed param)
            if strategy_name == "random":
                agent_makespan = runner(tasks_list, num_workers, seed=seed)
            else:
                agent_makespan = runner(tasks_list, num_workers)

            score = grade_episode(agent_makespan, optimal)

            results.append(BaselineResult(
                task_name=task_name,
                score=round(score, 4),
                agent_makespan=agent_makespan,
                optimal_makespan=optimal,
                agent_strategy=strategy_name,
            ))

        avg_score = sum(r.score for r in results) / len(results) if results else 0.0

        return BaselineResponse(
            results=results,
            average_score=round(avg_score, 4),
        )

    # ─────────────────────────────────────────────────────────
    # _build_observation() → SchedulerObservation (private)
    # ─────────────────────────────────────────────────────────

    def _build_observation(self) -> SchedulerObservation:
        """
        Construct a SchedulerObservation from the current simulation state.

        This translates between the core engine's dataclasses and the
        Pydantic models that are serialized over HTTP.
        """
        sim = self._sim
        assert sim is not None

        # Build TaskInfo for all tasks
        all_task_infos = [
            TaskInfo(
                task_id=t.task_id,
                duration=t.duration,
                dependencies=list(t.dependencies),
            )
            for t in self._tasks_list
        ]

        # Build TaskInfo for ready tasks only
        ready_task_ids = sim.get_ready_tasks()
        ready_task_infos = [
            TaskInfo(
                task_id=t.task_id,
                duration=t.duration,
                dependencies=list(t.dependencies),
            )
            for t in self._tasks_list
            if t.task_id in ready_task_ids
        ]

        # Build WorkerInfo
        worker_infos = [
            WorkerInfo(
                worker_id=w.worker_id,
                is_idle=(w.current_task is None),
                current_task=w.current_task,
                finish_time=w.finish_time if w.current_task is not None else None,
            )
            for w in sim.state.workers
        ]

        # Idle worker IDs
        idle_worker_ids = sim.get_idle_workers()

        return SchedulerObservation(
            current_time=sim.state.current_time,
            ready_tasks=ready_task_infos,
            idle_workers=idle_worker_ids,
            workers=worker_infos,
            running_tasks=dict(sim.state.running_tasks),
            completed_tasks=sorted(sim.state.completed_tasks),
            pending_tasks=sorted(sim.state.pending_tasks),
            all_tasks=all_task_infos,
            num_workers=self._num_workers,
            is_done=sim.state.is_done,
            last_action_valid=self._last_action_valid,
            message=self._last_message,
        )
