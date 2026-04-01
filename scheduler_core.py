"""
Enterprise Task Dependency Scheduler — Phase 1: Core Engine
============================================================
This module contains ALL the backend logic for the scheduling environment.

KEY INSIGHT FOR SCORE VARIANCE:
  For different scheduling strategies to produce different makespans,
  there must be a moment where:
    (a) More ready tasks exist than idle workers (CONTENTION), AND
    (b) Which task you pick for the limited workers MATTERS — i.e.,
        one choice leads to a longer total schedule than another.

  This means our DAG design MUST create "bottleneck moments" where
  the agent's decision genuinely affects the outcome.
"""

from __future__ import annotations

import copy
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────
# 1. DATA STRUCTURES — What is a "Task"?
# ─────────────────────────────────────────────────────────────


@dataclass
class Task:
    """
    Represents a single job to be scheduled on a worker.

    Attributes
    ----------
    task_id : str
        Unique identifier, e.g. "A", "build_frontend".
    duration : int
        How many time-steps this task takes to complete (>= 1).
    dependencies : list[str]
        List of task_ids that MUST finish before this task can start.
        An empty list means the task is "ready" immediately.
    """
    task_id: str
    duration: int
    dependencies: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        deps = ", ".join(self.dependencies) if self.dependencies else "none"
        return f"Task({self.task_id}, dur={self.duration}, deps=[{deps}])"


# ─────────────────────────────────────────────────────────────
# 2. SCENARIO DEFINITIONS — Easy, Medium, Hard
# ─────────────────────────────────────────────────────────────
#
# DESIGN PRINCIPLE: Each scenario is built to create at least
# one "bottleneck decision point" where more tasks are ready
# than workers are available. The WRONG choice at this point
# provably increases the makespan.


def create_easy_scenario(seed: Optional[int] = None) -> tuple[list[Task], int]:
    """
    EASY: Fork-join with 3 branches, only 2 workers
    ─────────────────────────────────────────────────
    Structure (7 tasks, 2 workers):

            ┌──→ fast1 (dur≈2) ──→ fast2 (dur≈2) ──┐
    start ──┤                                       ├──→ join
            ├──→ slow  (dur≈8)  ────────────────────┤
            └──→ mid   (dur≈4)  ────────────────────┘

    After "start" finishes, 3 tasks become ready: fast1, slow, mid.
    But we only have 2 workers. The agent must pick 2 of 3.

    OPTIMAL: Start "slow" and "mid" (or "slow" and "fast1").
      "slow" is on the critical path — it takes the longest.
      Starting it ASAP minimizes the makespan.

    SUBOPTIMAL: Start "fast1" + "mid" first, delay "slow".
      "slow" starts later → the join point waits longer → worse makespan.

    Alphabetical order picks: fast1, mid → DELAYS slow → SUBOPTIMAL! ✓
    """
    rng = random.Random(seed)

    task_defs = [
        ("start", 1, []),
        ("fast1", 2, ["start"]),
        ("fast2", 2, ["fast1"]),
        ("slow",  8, ["start"]),       # critical path! but sorts AFTER fast
        ("mid",   4, ["start"]),
        ("join",  1, ["fast2", "slow", "mid"]),
        ("done",  1, ["join"]),
    ]

    tasks = []
    for tid, base_dur, deps in task_defs:
        jitter = rng.randint(0, 2)
        tasks.append(Task(task_id=tid, duration=base_dur + jitter, dependencies=deps))

    num_workers = 2
    return tasks, num_workers


def create_medium_scenario(seed: Optional[int] = None) -> tuple[list[Task], int]:
    """
    MEDIUM: Two stages of contention with 2 workers
    ─────────────────────────────────────────────────
    Structure (10 tasks, 2 workers):

    Stage 1: init → (alpha, beta, gamma)    [3 ready, 2 workers — CONTENTION]
    Stage 2: alpha→a_next, beta→b_next      [unlocked by stage 1]
             gamma→g_next
    Merge:   merge ← [a_next, b_next, g_next]
    Final:   finish ← merge

    KEY CONTENTION:
    - alpha (dur≈2), beta (dur≈7), gamma (dur≈3)
    - "beta" is the LONGEST but sorts SECOND alphabetically
    - Critical path goes through beta → b_next → merge → finish
    - Picking alpha+gamma first (alphabetical) DELAYS beta

    With 2 workers and 3 competing branches, the agent MUST prioritize
    the critical path (beta) or suffer a longer makespan.
    """
    rng = random.Random(seed)

    task_defs = [
        ("init",    1, []),
        ("alpha",   2, ["init"]),        # SHORT, sorts FIRST
        ("beta",    7, ["init"]),        # LONG (critical!), sorts SECOND
        ("gamma",   3, ["init"]),        # MEDIUM, sorts THIRD
        ("a_next",  2, ["alpha"]),
        ("b_next",  2, ["beta"]),        # short after long critical
        ("g_next",  2, ["gamma"]),
        ("merge",   2, ["a_next", "b_next", "g_next"]),
        ("finish",  1, ["merge"]),
    ]

    tasks = []
    for tid, base_dur, deps in task_defs:
        jitter = rng.randint(0, 3)
        tasks.append(Task(task_id=tid, duration=base_dur + jitter, dependencies=deps))

    num_workers = 2
    return tasks, num_workers


def create_hard_scenario(seed: Optional[int] = None) -> tuple[list[Task], int]:
    """
    HARD: Multi-phase pipeline, 3 workers, heavy contention
    ────────────────────────────────────────────────────────
    15 tasks, 3 workers. Multiple bottleneck decisions.

    Phase 1: boot (root)
    Phase 2: 4 parallel tasks → only 3 workers! (contention #1)
        d_compile (dur≈8)  — CRITICAL PATH, sorts after others
        a_lint    (dur≈3)  — quick, sorts first
        b_fetch   (dur≈4)  — medium
        c_scan    (dur≈3)  — quick

    Phase 3: Testing depends on Phase 2
        e_unit   ← [d_compile, b_fetch]   (dur≈5)
        f_integ  ← [d_compile, a_lint]    (dur≈6)
        g_e2e    ← [b_fetch]              (dur≈7, LONG)
        h_perf   ← [c_scan]              (dur≈4)

    Phase 4: After Phase 2 completes, 4 tests ready, 3 workers (contention #2)
        i_bundle ← [e_unit, f_integ]      (dur≈3)
        j_docs   ← [f_integ, g_e2e]       (dur≈2)

    Phase 5:
        k_stage  ← [i_bundle]             (dur≈4)
        l_canary ← [i_bundle, j_docs]    (dur≈3)

    Phase 6:
        m_deploy ← [k_stage, l_canary]   (dur≈2)

    CONTENTION POINTS:
    1. After "boot": a_lint, b_fetch, c_scan, d_compile are ready.
       Only 3 workers. Alphabetical picks a_lint, b_fetch, c_scan → DELAYS d_compile!
       d_compile is critical (dur≈8). Starting it late is catastrophic.

    2. After d_compile finishes: e_unit, f_integ may become ready alongside
       g_e2e and h_perf. With 3 workers and 4 tasks, wrong picks matter.
    """
    rng = random.Random(seed)

    task_defs = [
        # Phase 1: Boot
        ("boot",      1, []),
        # Phase 2: Build tasks — 4 tasks, only 3 workers
        ("a_lint",    3, ["boot"]),        # quick, alphabetically FIRST
        ("b_fetch",   4, ["boot"]),        # medium
        ("c_scan",    3, ["boot"]),        # quick
        ("d_compile", 8, ["boot"]),        # CRITICAL PATH — sorts LAST!
        # Phase 3: Test tasks
        ("e_unit",    5, ["d_compile", "b_fetch"]),
        ("f_integ",   6, ["d_compile", "a_lint"]),
        ("g_e2e",     7, ["b_fetch"]),     # long but independent
        ("h_perf",    4, ["c_scan"]),
        # Phase 4: Package
        ("i_bundle",  3, ["e_unit", "f_integ"]),
        ("j_docs",    2, ["f_integ", "g_e2e"]),
        # Phase 5: Deploy
        ("k_stage",   4, ["i_bundle"]),
        ("l_canary",  3, ["i_bundle", "j_docs"]),
        # Phase 6: Release
        ("m_deploy",  2, ["k_stage", "l_canary"]),
    ]

    tasks = []
    for tid, base_dur, deps in task_defs:
        jitter = rng.randint(0, 4)  # large jitter for variance
        tasks.append(Task(task_id=tid, duration=base_dur + jitter, dependencies=deps))

    num_workers = 3
    return tasks, num_workers


# ─────────────────────────────────────────────────────────────
# 3. SIMULATION ENGINE — Running the Schedule
# ─────────────────────────────────────────────────────────────


@dataclass
class WorkerState:
    """State of a single worker/CPU core."""
    worker_id: int
    current_task: Optional[str] = None
    finish_time: int = 0


@dataclass
class SimulationState:
    """Complete snapshot of the simulation at any point in time."""
    current_time: int = 0
    workers: list[WorkerState] = field(default_factory=list)
    completed_tasks: set[str] = field(default_factory=set)
    running_tasks: dict[str, int] = field(default_factory=dict)
    pending_tasks: set[str] = field(default_factory=set)
    total_time: int = 0

    @property
    def is_done(self) -> bool:
        return len(self.pending_tasks) == 0 and len(self.running_tasks) == 0


class SchedulerSimulator:
    """
    The core simulation engine. Discrete-event simulation.

    Usage:
        sim = SchedulerSimulator(tasks, num_workers)
        sim.reset()
        while not sim.state.is_done:
            ready = sim.get_ready_tasks()
            idle = sim.get_idle_workers()
            if ready and idle:
                sim.assign_task(ready[0], idle[0])
            else:
                sim.advance_time()
        result = sim.state.total_time
    """

    def __init__(self, tasks: list[Task], num_workers: int):
        self.tasks = {t.task_id: t for t in tasks}
        self.num_workers = num_workers
        self.state = SimulationState()

    def reset(self) -> SimulationState:
        self.state = SimulationState(
            current_time=0,
            workers=[WorkerState(worker_id=i) for i in range(self.num_workers)],
            completed_tasks=set(),
            running_tasks={},
            pending_tasks=set(self.tasks.keys()),
            total_time=0,
        )
        return self.state

    def get_ready_tasks(self) -> list[str]:
        """Return pending tasks whose dependencies are all completed."""
        ready = []
        for task_id in self.state.pending_tasks:
            task = self.tasks[task_id]
            if all(dep in self.state.completed_tasks for dep in task.dependencies):
                ready.append(task_id)
        return sorted(ready)

    def get_idle_workers(self) -> list[int]:
        return [w.worker_id for w in self.state.workers if w.current_task is None]

    def assign_task(self, task_id: str, worker_id: int) -> bool:
        """Assign a ready task to an idle worker. Returns success."""
        if task_id not in self.state.pending_tasks:
            return False
        if task_id not in self.get_ready_tasks():
            return False
        worker = self.state.workers[worker_id]
        if worker.current_task is not None:
            return False

        task = self.tasks[task_id]
        finish_time = self.state.current_time + task.duration
        worker.current_task = task_id
        worker.finish_time = finish_time
        self.state.pending_tasks.remove(task_id)
        self.state.running_tasks[task_id] = finish_time
        return True

    def advance_time(self) -> int:
        """Jump to the next task completion event. Returns new time or -1."""
        if not self.state.running_tasks:
            return -1

        next_finish = min(self.state.running_tasks.values())
        self.state.current_time = next_finish

        newly_completed = [
            tid for tid, ft in self.state.running_tasks.items()
            if ft <= next_finish
        ]
        for tid in newly_completed:
            self.state.completed_tasks.add(tid)
            del self.state.running_tasks[tid]
            for w in self.state.workers:
                if w.current_task == tid:
                    w.current_task = None
                    w.finish_time = 0
                    break

        if self.state.is_done:
            self.state.total_time = self.state.current_time

        return self.state.current_time


# ─────────────────────────────────────────────────────────────
# 4. OPTIMAL SOLVER — Critical Path heuristic
# ─────────────────────────────────────────────────────────────


def compute_critical_path_length(tasks: dict[str, Task]) -> dict[str, int]:
    """
    Compute the longest-path distance from each task to the DAG's end.
    Returns: task_id → critical path length from that task forward.
    """
    memo: dict[str, int] = {}

    def dfs(task_id: str) -> int:
        if task_id in memo:
            return memo[task_id]
        task = tasks[task_id]
        successors = [t for t in tasks.values() if task_id in t.dependencies]
        if not successors:
            memo[task_id] = task.duration
        else:
            memo[task_id] = task.duration + max(dfs(s.task_id) for s in successors)
        return memo[task_id]

    for tid in tasks:
        dfs(tid)
    return memo


def _run_with_policy(tasks_list: list[Task], num_workers: int,
                     priority_fn, verbose: bool = False) -> int:
    """
    Run simulation with a priority function.
    priority_fn(task_id, tasks_dict, cp_lengths) → sortable value (lower = first).
    """
    tasks = {t.task_id: t for t in tasks_list}
    cp_lengths = compute_critical_path_length(tasks)
    sim = SchedulerSimulator(tasks_list, num_workers)
    sim.reset()

    while not sim.state.is_done:
        ready = sim.get_ready_tasks()
        idle = sim.get_idle_workers()

        if ready and idle:
            ready.sort(key=lambda tid: priority_fn(tid, tasks, cp_lengths))
            for task_id in ready:
                if not idle:
                    break
                worker_id = idle.pop(0)
                if verbose:
                    print(f"    t={sim.state.current_time:3d}: "
                          f"assign {task_id} → worker {worker_id}")
                sim.assign_task(task_id, worker_id)
        else:
            sim.advance_time()

    return sim.state.total_time


def compute_optimal_makespan(tasks_list: list[Task], num_workers: int) -> int:
    """Optimal (heuristic): schedule by longest remaining critical path."""
    return _run_with_policy(
        tasks_list, num_workers,
        priority_fn=lambda tid, tasks, cp: (-cp[tid], tid)
    )


# ─────────────────────────────────────────────────────────────
# 5. AGENT STRATEGIES — Show score variance
# ─────────────────────────────────────────────────────────────


def run_alphabetical_agent(tasks_list: list[Task], num_workers: int,
                           verbose: bool = False) -> int:
    """Picks alphabetically first ready task. Suboptimal on our DAGs."""
    return _run_with_policy(
        tasks_list, num_workers,
        priority_fn=lambda tid, tasks, cp: tid,
        verbose=verbose
    )


def run_shortest_first_agent(tasks_list: list[Task], num_workers: int) -> int:
    """Picks shortest duration task first. Ignores critical path."""
    return _run_with_policy(
        tasks_list, num_workers,
        priority_fn=lambda tid, tasks, cp: (tasks[tid].duration, tid)
    )


def run_longest_first_agent(tasks_list: list[Task], num_workers: int) -> int:
    """Picks longest duration task first. Better but not path-aware."""
    return _run_with_policy(
        tasks_list, num_workers,
        priority_fn=lambda tid, tasks, cp: (-tasks[tid].duration, tid)
    )


def run_random_agent(tasks_list: list[Task], num_workers: int,
                     seed: int = 0) -> int:
    """Random task selection each step."""
    rng = random.Random(seed)
    sim = SchedulerSimulator(tasks_list, num_workers)
    sim.reset()

    while not sim.state.is_done:
        ready = sim.get_ready_tasks()
        idle = sim.get_idle_workers()
        if ready and idle:
            rng.shuffle(ready)
            for task_id in ready:
                if not idle:
                    break
                worker_id = idle.pop(0)
                sim.assign_task(task_id, worker_id)
        else:
            sim.advance_time()

    return sim.state.total_time


# ─────────────────────────────────────────────────────────────
# 6. GRADER — Deterministic Scoring (0.0 to 1.0)
# ─────────────────────────────────────────────────────────────


def grade_episode(agent_makespan: int, optimal_makespan: int) -> float:
    """
    Score = optimal_time / agent_time, clamped to [0.0, 1.0].
    Perfect play → 1.0. Twice as slow → 0.5.
    """
    if agent_makespan <= 0:
        return 0.0
    return max(0.0, min(1.0, optimal_makespan / agent_makespan))


# ─────────────────────────────────────────────────────────────
# 7. DEMO — Prove variance
# ─────────────────────────────────────────────────────────────


def demo_scenario(name: str, create_fn, seed: int = 42):
    """Run all agents, print results, verify score variance."""
    tasks, num_workers = create_fn(seed=seed)

    print(f"\n{'='*72}")
    print(f"  SCENARIO: {name}")
    print(f"  Tasks: {len(tasks)} | Workers: {num_workers} | Seed: {seed}")
    print(f"{'='*72}")

    # Print task graph
    print("\n  Task Graph:")
    for t in tasks:
        deps = ", ".join(t.dependencies) if t.dependencies else "(root)"
        print(f"    {t.task_id:12s}  dur={t.duration:2d}  deps=[{deps}]")

    # Run all agent strategies
    optimal     = compute_optimal_makespan(tasks, num_workers)
    alphabetic  = run_alphabetical_agent(tasks, num_workers)
    shortest    = run_shortest_first_agent(tasks, num_workers)
    longest     = run_longest_first_agent(tasks, num_workers)
    rand_times  = [run_random_agent(tasks, num_workers, seed=s) for s in range(5)]

    print(f"\n  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │  Strategy            │ Makespan │ Score │ Visual        │")
    print(f"  ├─────────────────────────────────────────────────────────┤")

    all_scores = []
    results = [
        ("Critical Path (opt)", optimal),
        ("Alphabetical", alphabetic),
        ("Shortest First", shortest),
        ("Longest First", longest),
        *[(f"Random (seed={i})", t) for i, t in enumerate(rand_times)],
    ]

    for label, makespan in results:
        score = grade_episode(makespan, optimal)
        all_scores.append(score)
        bar = "█" * int(score * 15) + "░" * (15 - int(score * 15))
        print(f"  │  {label:20s} │  {makespan:5d}   │ {score:.3f} │ {bar} │")

    print(f"  └─────────────────────────────────────────────────────────┘")

    unique_scores = len(set(round(s, 3) for s in all_scores))
    if unique_scores > 1:
        print(f"  ✅ VARIANCE: {unique_scores} distinct scores across strategies")
    else:
        print(f"  ⚠️  All strategies scored identically (no contention?)")

    # Show variance across episode seeds
    print(f"\n  Variance across episode seeds (Alphabetical agent):")
    ep_scores = set()
    for ep_seed in range(8):
        t, w = create_fn(seed=ep_seed)
        opt = compute_optimal_makespan(t, w)
        naive = run_alphabetical_agent(t, w)
        sc = grade_episode(naive, opt)
        ep_scores.add(round(sc, 3))
        marker = ""
        if round(sc, 3) < 1.0:
            marker = " ← suboptimal!"
        print(f"    seed={ep_seed}: opt={opt:2d}, naive={naive:2d}, "
              f"score={sc:.3f}{marker}")

    if len(ep_scores) > 1:
        print(f"  ✅ SEED VARIANCE: {len(ep_scores)} distinct scores")
    else:
        print(f"  ⚠️  Same score across all seeds")

    # Trace one episode to show the exact scheduling decisions
    if name.startswith("Hard") or name.startswith("Easy"):
        print(f"\n  Trace — Alphabetical agent decisions (seed={seed}):")
        run_alphabetical_agent(tasks, num_workers, verbose=True)

    return optimal


if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  Enterprise Task Dependency Scheduler — Phase 1 Demo            ║")
    print("║  Data structures • Simulation • Grading • Variance proof        ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    demo_scenario("Easy  (Fork-Join, 2 workers)", create_easy_scenario, seed=42)
    demo_scenario("Medium (3 Branches, 2 workers)", create_medium_scenario, seed=42)
    demo_scenario("Hard  (CI/CD Pipeline, 3 workers)", create_hard_scenario, seed=42)

    print(f"\n{'='*72}")
    print("  Summary:")
    print("   • Task: dataclass with task_id, duration, dependencies")
    print("   • SimulationEngine: assign_task() + advance_time() loop")
    print("   • Optimal Solver: Critical Path Length heuristic")
    print("   • Grader: score = optimal / agent_time ∈ [0.0, 1.0]")
    print("   • Variance: Different strategies → different scores ✓")
    print("   • Variance: Different seeds → different scores ✓")
    print(f"{'='*72}")
