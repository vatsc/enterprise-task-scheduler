---
title: Enterprise Task Scheduler
emoji: 🏗️
colorFrom: blue
colorTo: green
sdk: docker
tags:
  - openenv
---

# 🏗️ Enterprise Task Dependency Scheduler

> An **OpenEnv** environment where an AI agent schedules dependent tasks across multiple workers to minimize total execution time (makespan).

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/openenv-dev/openenv)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)](https://fastapi.tiangolo.com/)

## 🎯 Overview

The agent acts as a **dynamic load balancer** for a high-performance computing cluster. It must schedule jobs with complex dependencies across multiple workers (CPU cores) to minimize total execution time — a real-world problem faced by CI/CD systems, build tools, and HPC clusters.

```
     ┌──→ fast1 ──→ fast2 ──┐
     │                       │
start──→ slow (CRITICAL!) ───┼──→ join ──→ done
     │                       │
     └──→ mid ───────────────┘
     
     2 workers, 3 ready tasks → CONTENTION!
     Wrong choice at this point ≈ 20% worse makespan.
```

## 📋 Tasks (Difficulty Levels)

| Task | # Jobs | Workers | Key Challenge | Optimal Approx. |
|------|--------|---------|---------------|-----------------|
| **Easy** | 7 | 2 | Fork-join with 1 contention point | ~15 steps |
| **Medium** | 9 | 2 | Two-stage contention, 3 branches | ~17 steps |
| **Hard** | 14 | 3 | Multi-phase CI/CD pipeline | ~35 steps |

## 🔌 API

### Standard OpenEnv Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Execute one action |
| `GET` | `/state` | Get episode metadata |

### Custom Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/grader` | Get scoring info (0.0–1.0) |
| `POST` | `/baseline` | Run built-in agent strategies |
| `GET` | `/tasks` | List tasks and schemas |
| `GET` | `/health` | Health check |

## 🎮 Action Space

The agent sends **one action per step**:

```json
// Assign a task to a worker
{"action_type": "assign", "task_id": "d_compile", "worker_id": 0}

// Wait for running tasks to finish
{"action_type": "wait"}
```

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `"assign" \| "wait"` | What to do |
| `task_id` | `string` | Task to schedule (required for assign) |
| `worker_id` | `int` | Worker to assign to (required for assign) |

## 👁️ Observation Space

After each action, the agent sees:

```json
{
  "current_time": 3,
  "ready_tasks": [{"task_id": "slow", "duration": 10, "dependencies": ["start"]}],
  "idle_workers": [0, 1],
  "workers": [{"worker_id": 0, "is_idle": true, "current_task": null}],
  "running_tasks": {"fast1": 5},
  "completed_tasks": ["start"],
  "pending_tasks": ["fast2", "join", "done"],
  "all_tasks": [...],
  "num_workers": 2,
  "is_done": false,
  "last_action_valid": true,
  "message": "Advanced time from t=0 to t=3."
}
```

## 📊 Grading

```
score = optimal_makespan / agent_makespan ∈ [0.0, 1.0]
```

- **1.0** = perfect (matched the optimal solver)
- **0.5** = twice as slow as optimal
- Score varies with seeds AND strategies (verified)

### Baseline Scores (seed=42)

| Strategy | Easy | Medium | Hard |
|----------|------|--------|------|
| Critical Path (optimal) | 1.000 | 1.000 | 1.000 |
| Longest First | 1.000 | 1.000 | 1.000 |
| Random | 0.882 | 0.905 | varies |
| Alphabetical | 0.789 | 1.000 | 0.969 |
| Shortest First | 0.789 | 0.792 | 0.969 |

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
pip install fastapi uvicorn pydantic httpx

# Start the server
python3 -m uvicorn server.app:app --port 8000

# Test it
curl http://localhost:8000/health
curl http://localhost:8000/tasks
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" \
     -d '{"task_name": "easy", "seed": 42}'
```

### Run the Baseline Agent

```bash
# Greedy heuristic (no API key needed)
python3 inference.py --no-llm

# With OpenAI LLM
export OPENAI_API_KEY="sk-..."
python3 inference.py --model gpt-4o-mini
```

### Docker

```bash
docker build -t scheduler-env .
docker run -p 8000:8000 scheduler-env
```

## 📁 Project Structure

```
scheduler_env/
├── scheduler_core.py     # Phase 1: simulation engine, scenarios, grader
├── models.py             # Pydantic models (Action, Observation, State, etc.)
├── environment.py        # OpenEnv wrapper (reset/step/state)
├── inference.py          # Baseline agent (LLM + greedy fallback)
├── openenv.yaml          # Environment manifest
├── pyproject.toml        # Python project config
├── Dockerfile            # Docker build
├── README.md             # This file
├── __init__.py           # Package exports
└── server/
    ├── app.py            # FastAPI server (7 endpoints)
    └── requirements.txt  # Server dependencies
```

## 🧠 Strategy Guide

The key to optimal scheduling is **critical path awareness**:

1. **Identify the critical path** — the longest chain of dependent tasks
2. **At contention points** (more ready tasks than workers), always prioritize tasks on the critical path
3. **Never delay a long task** in favor of a short one, even if the short one sorts first alphabetically

Example: In the Easy scenario, `slow` (duration≈8) is on the critical path but sorts after `fast1` alphabetically. An alphabetical agent delays `slow` → 20% worse makespan.

## 📄 License

MIT
