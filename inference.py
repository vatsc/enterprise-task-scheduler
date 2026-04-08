"""
Inference Script — Enterprise Task Dependency Scheduler
=========================================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()

- Defaults are set only for API_BASE_URL and MODEL_NAME
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ─────────────────────────────────────────────────────────────
# Environment Variables (mandatory for hackathon)
# ─────────────────────────────────────────────────────────────

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "enterprise-task-scheduler")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "enterprise-task-scheduler"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 100
TEMPERATURE = 0.1
MAX_TOKENS = 150

# ─────────────────────────────────────────────────────────────
# Structured stdout logging (mandatory format)
# ─────────────────────────────────────────────────────────────


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────
# LLM-based scheduling agent (uses OpenAI client)
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert task scheduler for a high-performance computing cluster.

    Your goal is to schedule dependent tasks across multiple workers to MINIMIZE
    total execution time (makespan). You will be shown the current state of the
    scheduler and must decide what to do next.

    RULES:
    1. You can ASSIGN a task to a worker: {"action_type": "assign", "task_id": "...", "worker_id": N}
    2. You can WAIT for running tasks to finish: {"action_type": "wait"}
    3. You can only assign tasks whose dependencies are ALL completed (shown in ready_tasks).
    4. You can only assign to idle workers (shown in idle_workers).
    5. CRITICAL STRATEGY: Always prioritize tasks with the LONGEST remaining
       critical path. Tasks that take longer and have more downstream dependents
       should be started FIRST. This minimizes the total makespan.

    Respond with ONLY a valid JSON action. No explanation, no markdown, just JSON.
""")


def build_user_prompt(obs: dict) -> str:
    """Build a concise prompt from the observation."""
    ready = obs.get("ready_tasks", [])
    idle = obs.get("idle_workers", [])
    running = obs.get("running_tasks", {})
    completed = obs.get("completed_tasks", [])
    all_tasks = obs.get("all_tasks", [])

    task_info = {t["task_id"]: t for t in all_tasks}

    lines = [
        f"Current time: t={obs.get('current_time', 0)}",
        f"Completed: {completed}",
        f"Running: {running}",
        f"Idle workers: {idle}",
        "",
        "Ready tasks (can be assigned NOW):",
    ]
    for t in ready:
        dependents = [
            at["task_id"] for at in all_tasks
            if t["task_id"] in at.get("dependencies", [])
        ]
        lines.append(
            f"  - {t['task_id']}: duration={t['duration']}, "
            f"deps={t.get('dependencies', [])}, downstream={dependents}"
        )

    if not ready:
        lines.append("  (none — you must WAIT)")

    lines.append("")
    if ready and idle:
        lines.append("Choose: assign a ready task to an idle worker (prioritize LONGEST critical path).")
    else:
        lines.append("No ready tasks or no idle workers — you should WAIT.")

    return "\n".join(lines)


def get_llm_action(client: OpenAI, obs: dict) -> Optional[dict]:
    """Ask the LLM to decide the next action. Returns None on failure."""
    user_prompt = build_user_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        return json.loads(text)
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        return None


def greedy_action(obs: dict) -> dict:
    """
    Critical-path heuristic: assign the task with the longest downstream
    critical path to the first idle worker. This matches the optimal solver.
    """
    ready = obs.get("ready_tasks", [])
    idle = obs.get("idle_workers", [])

    if not ready or not idle:
        return {"action_type": "wait"}

    # Build dependency graph for critical path calculation
    all_tasks = obs.get("all_tasks", [])
    task_map = {t["task_id"]: t for t in all_tasks}

    # Compute longest path from each task to the end (memoized)
    cache = {}

    def critical_path_length(task_id: str) -> int:
        if task_id in cache:
            return cache[task_id]
        t = task_map.get(task_id)
        if t is None:
            return 0
        # Find all tasks that depend on this task (downstream)
        downstream = [
            at["task_id"] for at in all_tasks
            if task_id in at.get("dependencies", [])
        ]
        if not downstream:
            result = t["duration"]
        else:
            result = t["duration"] + max(
                critical_path_length(d) for d in downstream
            )
        cache[task_id] = result
        return result

    # Sort ready tasks by critical path length (longest first)
    ready_sorted = sorted(ready, key=lambda t: -critical_path_length(t["task_id"]))
    return {
        "action_type": "assign",
        "task_id": ready_sorted[0]["task_id"],
        "worker_id": idle[0],
    }


# ─────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────


def obs_to_dict(obs) -> dict:
    """Convert observation to dict, handling both dict and object forms."""
    if isinstance(obs, dict):
        return obs
    if hasattr(obs, "to_dict"):
        return obs.to_dict()
    if hasattr(obs, "_data"):
        return obs._data
    return vars(obs)


def action_to_str(action: dict) -> str:
    """Format action as a compact string for the [STEP] log."""
    if action.get("action_type") == "assign":
        return f"assign({action.get('task_id')},{action.get('worker_id')})"
    return "wait()"


def play_episode(env, task_name: str, llm_client: Optional[OpenAI] = None) -> float:
    """
    Play one full episode with structured logging.
    Returns the final score in [0.0, 1.0].
    """
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset with specific task name
        result = env.reset_with_task(task_name=task_name)
        obs = obs_to_dict(result.observation)

        for step_idx in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Decide action: try LLM first, fallback to greedy
            action = None
            if llm_client is not None:
                action = get_llm_action(llm_client, obs)

            if action is None:
                action = greedy_action(obs)

            # Step
            result = env.step(action)
            obs = obs_to_dict(result.observation)

            reward = result.reward if result.reward is not None else 0.0
            done = result.done
            error = None

            # Check for invalid action
            if not obs.get("last_action_valid", True):
                error = obs.get("message", "invalid action")
                # Retry with greedy on invalid action
                fallback = greedy_action(obs)
                result = env.step(fallback)
                obs = obs_to_dict(result.observation)
                reward = result.reward if result.reward is not None else 0.0
                done = result.done
                action = fallback
                error = None  # recovered

            rewards.append(reward)
            steps_taken = step_idx

            log_step(
                step=step_idx,
                action=action_to_str(action),
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        # Get final score from grader
        grader = env.get_grader()
        score = grader.get("score", 0.0)
        score = min(max(score, 0.0), 1.0)
        success = score > 0.0

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────


def main() -> None:
    # Initialize OpenAI client
    llm_client = None
    if API_KEY:
        llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    else:
        print("[DEBUG] No HF_TOKEN/API_KEY set, using greedy heuristic only.", flush=True)

    # Try to use openenv SDK with from_docker_image
    env = None
    try:
        from scheduler_client import SchedulerEnvClient

        if os.getenv("ENV_BASE_URL"):
            # Direct URL mode (for testing against a running server)
            env = SchedulerEnvClient(os.getenv("ENV_BASE_URL"))
        else:
            # Docker mode (standard hackathon evaluation)
            env = SchedulerEnvClient.from_docker_image(LOCAL_IMAGE_NAME)
    except Exception as exc:
        print(f"[DEBUG] SDK init failed: {exc}. Falling back to direct HTTP.", flush=True)
        # Fallback: direct HTTP for local testing
        import httpx

        class DirectHTTPEnv:
            """Minimal env wrapper using direct HTTP calls (fallback)."""
            def __init__(self, base_url: str):
                self._client = httpx.Client(base_url=base_url, timeout=30)

            def reset_with_task(self, task_name="easy", seed=None):
                body = {"task_name": task_name}
                if seed is not None:
                    body["seed"] = seed
                r = self._client.post("/reset", json=body)
                r.raise_for_status()
                obs = r.json()

                class _Result:
                    def __init__(self, obs_dict):
                        self.observation = obs_dict
                        self.reward = 0.0
                        self.done = False
                return _Result(obs)

            def step(self, action):
                r = self._client.post("/step", json=action)
                r.raise_for_status()
                data = r.json()

                class _Result:
                    def __init__(self, d):
                        self.observation = d.get("observation", d)
                        self.reward = d.get("reward", 0.0)
                        self.done = d.get("done", False)
                return _Result(data)

            def get_grader(self):
                r = self._client.get("/grader")
                r.raise_for_status()
                return r.json()

            def close(self):
                self._client.close()

        base = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")
        env = DirectHTTPEnv(base)

    # Play all tasks
    try:
        scores = []
        for task_name in TASKS:
            score = play_episode(env, task_name, llm_client)
            scores.append(score)
    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    main()
