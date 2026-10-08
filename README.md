# Terminal Coding Agent

A clean, modular, self-improving terminal coding agent built from scratch.

## Architecture

```
src/coding_agent/
├── agent.py              # Core agent loop (LLM + tools + memory retrieval)
├── cli.py                # CLI entry point with pipeline integration
├── client.py             # LLM HTTP client
├── config.py             # Config loading from env
├── evaluator.py          # Deterministic trajectory evaluator
├── memory.py             # Long-term experience memory (local JSON, no vector DB)
├── models.py             # Shared data models (AgentResult, ModelResponse, …)
├── parser.py             # Response / tool-call parser
├── pipeline.py           # Post-run pipeline: evaluate → reflect → persist → store
├── reflection.py         # LLM-based structured reflection generator
├── registry.py           # Tool registry
├── run_record.py         # Execution trajectory recorder
├── tools.py              # Built-in tools (shell, read/write/edit file)
├── trajectory_store.py   # Local JSON trajectory persistence
└── weakness_analyzer.py  # Corpus-level recurring weakness detection
```

## Self-Improvement Loop

Every completed run goes through the **ExperiencePipeline** automatically:

1. **Evaluate** — `TrajectoryEvaluator` scores the run deterministically (completion, tool failures, verification commands).
2. **Reflect** — `ReflectionGenerator` asks an LLM to produce a structured retrospective (`what_worked`, `what_failed`, `why_it_failed`, `what_to_do_differently`).
3. **Persist** — `TrajectoryStore` saves the full trajectory JSON with embedded evaluation and reflection.
4. **Store** — `ExperienceMemory` saves the experience for future retrieval.

Before each new run, the agent retrieves the most relevant past experiences (by token-overlap + evaluation score) and injects them as optional advisory context — without overriding system instructions.

## Weakness Analysis

`WeaknessAnalyzer` performs **corpus-level** analysis across a collection of evaluated trajectories to identify *recurring* behavioural problems. It is fully deterministic — no LLM calls — and never modifies the agent.

Six weakness patterns are detected:

| Pattern | Signal |
|---|---|
| `UNNECESSARY_TOOL_CALLS` | ≥10 total calls with ≥40% going to a single tool |
| `REPEATED_FAILED_APPROACHES` | Same tool errors ≥2 times in one run |
| `POOR_VERIFICATION` | Code written but no test/lint command run |
| `INEFFICIENT_PLANNING` | Steps > 2× corpus median (outlier detection) |
| `BAD_TOOL_SELECTION` | Shell used for `cat`/`echo`/`head` instead of `read_file` |
| `FAILURE_TO_RECOVER` | Identical tool args re-tried immediately after an error |

Each detected pattern produces a structured `ImprovementProposal` containing:
- **Observed problem** with corpus-wide statistics
- **Trajectory-level evidence** (run IDs, task snippets, metrics)
- **Likely cause** — root-cause hypothesis
- **Proposed improvement** — concrete system-prompt or architecture change
- **Expected benefit** — measurable outcome if applied
- **Risk** — potential downside or tradeoff

```python
from coding_agent import WeaknessAnalyzer, TrajectoryStore

store = TrajectoryStore()
records = store.list_records()

analyzer = WeaknessAnalyzer(min_evidence_count=2)
report = analyzer.analyse_records(records)

print(report.summary)
for proposal in report.proposals:
    print(f"[{proposal.severity.value.upper()}] {proposal.weakness_type.value}")
    print(f"  Problem:     {proposal.observed_problem}")
    print(f"  Fix:         {proposal.proposed_improvement}")
    print(f"  Prevalence:  {proposal.prevalence:.0%} of runs")
```

## Running the Application

```bash
python -m coding_agent                        # interactive REPL
python -m coding_agent "your task here"       # single non-interactive prompt
python -m coding_agent --plan "your task"     # plan mode
python -m coding_agent --auto-approve "task"  # skip write-tool approval prompts
python -m coding_agent --no-reflection "task" # skip LLM reflection
python -m coding_agent --no-memory "task"     # disable experience memory
python -m coding_agent --only-store-successful "task"  # save only successful runs
python -m coding_agent --verbose-pipeline "task"       # show pipeline status
```

Or install in editable mode first:

```bash
pip install -e .
coding-agent
```

## Running Tests

```bash
python -m pytest tests/ -v
```

248 tests covering the full agent stack: trajectory recording, deterministic evaluation, LLM reflection, experience memory, pipeline orchestration, weakness detection, CLI integration, and all core tools.
