"""
Baseline Inference Script — Enterprise Task Dependency Scheduler
=================================================================
This script plays the environment using an OpenAI-compatible LLM agent.
It connects to the running server, plays all 3 difficulty levels, and
reports scores.

Usage:
    # Start the server first:
    python3 -m uvicorn server.app:app --port 8000

    # Then run the agent:
    export OPENAI_API_KEY="your-key-here"
    python3 inference.py

    # Or with a custom server URL:
    python3 inference.py --base-url http://localhost:8000

    # Run without LLM (uses greedy heuristic as fallback):
    python3 inference.py --no-llm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx

# ─────────────────────────────────────────────────────────────
# LLM Agent — uses OpenAI API to make scheduling decisions
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
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
"""


def build_user_prompt(obs: dict) -> str:
    """Build a concise prompt from the observation."""
    ready = obs["ready_tasks"]
    idle = obs["idle_workers"]
    running = obs["running_tasks"]
    completed = obs["completed_tasks"]
    all_tasks = obs["all_tasks"]

    # Build dependency chains for context
    task_info = {}
    for t in all_tasks:
        task_info[t["task_id"]] = t

    lines = [
        f"Current time: t={obs['current_time']}",
        f"Completed: {completed}",
        f"Running: {running}",
        f"Idle workers: {idle}",
        "",
        "Ready tasks (can be assigned NOW):",
    ]
    for t in ready:
        # Count downstream dependents
        dependents = [
            at["task_id"] for at in all_tasks
            if t["task_id"] in at["dependencies"]
        ]
        lines.append(
            f"  - {t['task_id']}: duration={t['duration']}, "
            f"deps={t['dependencies']}, downstream={dependents}"
        )

    if not ready:
        lines.append("  (none — you must WAIT)")

    lines.append("")
    lines.append("Pending tasks (not yet ready):")
    for tid in obs["pending_tasks"]:
        if tid in [t["task_id"] for t in ready]:
            continue
        t = task_info.get(tid, {})
        lines.append(f"  - {tid}: duration={t.get('duration','?')}, deps={t.get('dependencies','?')}")

    lines.append("")
    if ready and idle:
        lines.append("Choose: assign a ready task to an idle worker (prioritize LONGEST critical path).")
    else:
        lines.append("No ready tasks or no idle workers — you should WAIT.")

    return "\n".join(lines)


def call_llm(system: str, user: str, api_key: str, model: str = "gpt-4o-mini") -> dict:
    """Call OpenAI API and parse the JSON response."""
    client = httpx.Client(timeout=30)
    resp = client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": 100,
        },
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    return json.loads(content)


# ─────────────────────────────────────────────────────────────
# Greedy Heuristic Agent — fallback when no LLM key is set
# ─────────────────────────────────────────────────────────────


def greedy_action(obs: dict) -> dict:
    """
    Simple heuristic: assign the LONGEST-duration ready task to the
    first idle worker. This approximates critical-path scheduling.
    """
    ready = obs["ready_tasks"]
    idle = obs["idle_workers"]

    if not ready or not idle:
        return {"action_type": "wait"}

    # Sort by duration descending (longest first = critical path heuristic)
    ready_sorted = sorted(ready, key=lambda t: -t["duration"])
    return {
        "action_type": "assign",
        "task_id": ready_sorted[0]["task_id"],
        "worker_id": idle[0],
    }


# ─────────────────────────────────────────────────────────────
# Episode Runner
# ─────────────────────────────────────────────────────────────


def play_episode(
    base_url: str,
    task_name: str,
    seed: int,
    use_llm: bool = False,
    api_key: str = "",
    model: str = "gpt-4o-mini",
    verbose: bool = True,
) -> dict:
    """Play one full episode and return the grader results."""
    client = httpx.Client(base_url=base_url, timeout=30)

    # Reset
    obs = client.post("/reset", json={"task_name": task_name, "seed": seed}).json()

    if verbose:
        print(f"\n  {'─' * 50}")
        print(f"  Episode: {task_name} (seed={seed})")
        print(f"  Tasks: {len(obs['all_tasks'])} | Workers: {obs['num_workers']}")
        print(f"  {'─' * 50}")

    steps = 0
    max_steps = 200
    result = None

    while not obs["is_done"] and steps < max_steps:
        # Decide action
        if use_llm and api_key:
            try:
                user_prompt = build_user_prompt(obs)
                action = call_llm(SYSTEM_PROMPT, user_prompt, api_key, model)
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  LLM error: {e}, falling back to greedy")
                action = greedy_action(obs)
        else:
            action = greedy_action(obs)

        # Step
        resp = client.post("/step", json=action)
        result = resp.json()
        obs = result["observation"]
        steps += 1

        if verbose and not obs["last_action_valid"]:
            print(f"  ⚠️  Step {steps}: Invalid action — {obs['message']}")
            # Retry with greedy on invalid action
            action = greedy_action(obs)
            resp = client.post("/step", json=action)
            result = resp.json()
            obs = result["observation"]
            steps += 1

        if result["done"]:
            break

    # Get grader info
    grader = client.get("/grader").json()

    if verbose:
        print(f"  ✅ Done in {steps} steps")
        print(f"  Score: {grader['score']:.3f} "
              f"(agent={grader['agent_makespan']}, optimal={grader['optimal_makespan']})")

    return {
        "task_name": task_name,
        "seed": seed,
        "steps": steps,
        "score": grader["score"],
        "agent_makespan": grader["agent_makespan"],
        "optimal_makespan": grader["optimal_makespan"],
    }


# ─────────────────────────────────────────────────────────────
# Main — Run baseline across all difficulties
# ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Baseline agent for Enterprise Task Scheduler"
    )
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000",
        help="Server URL (default: http://127.0.0.1:8000)"
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Use greedy heuristic instead of LLM"
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for episodes (default: 42)"
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["easy", "medium", "hard"],
        help="Difficulties to test (default: easy medium hard)"
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    use_llm = not args.no_llm and bool(api_key)

    agent_type = f"LLM ({args.model})" if use_llm else "Greedy Heuristic"

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  Enterprise Task Scheduler — Baseline Agent              ║")
    print(f"║  Agent: {agent_type:<48s}║")
    print("╚═══════════════════════════════════════════════════════════╝")

    if not use_llm and not args.no_llm:
        print("\n  ℹ️  No OPENAI_API_KEY set. Using greedy heuristic.")
        print("  Set OPENAI_API_KEY or use --no-llm to suppress this message.\n")

    # Check server is running
    try:
        r = httpx.get(f"{args.base_url}/health", timeout=5)
        assert r.status_code == 200
    except Exception:
        print(f"\n  ❌ Cannot reach server at {args.base_url}")
        print("  Start it first: python3 -m uvicorn server.app:app --port 8000")
        sys.exit(1)

    # Play all difficulties
    results = []
    for task_name in args.tasks:
        result = play_episode(
            base_url=args.base_url,
            task_name=task_name,
            seed=args.seed,
            use_llm=use_llm,
            api_key=api_key,
            model=args.model,
        )
        results.append(result)

    # Summary table
    print(f"\n  {'═' * 55}")
    print(f"  {'Task':<10} {'Score':>8} {'Agent':>8} {'Optimal':>8} {'Steps':>8}")
    print(f"  {'─' * 55}")
    for r in results:
        print(f"  {r['task_name']:<10} {r['score']:>8.3f} {r['agent_makespan']:>8} "
              f"{r['optimal_makespan']:>8} {r['steps']:>8}")
    avg = sum(r["score"] for r in results) / len(results)
    print(f"  {'─' * 55}")
    print(f"  {'AVERAGE':<10} {avg:>8.3f}")
    print(f"  {'═' * 55}")

    # Return results as JSON for the /baseline endpoint integration
    return results


if __name__ == "__main__":
    main()
