# Terminal Coding Agent

A clean, modular, self-improving terminal coding agent built from scratch.

## Architecture

```
src/coding_agent/
├── agent.py            # Core agent loop (LLM + tools + memory retrieval)
├── cli.py              # CLI entry point with pipeline integration
├── client.py           # LLM HTTP client
├── config.py           # Config loading from env
├── evaluator.py        # Deterministic trajectory evaluator
├── memory.py           # Long-term experience memory (local JSON, no vector DB)
├── models.py           # Shared data models (AgentResult, ModelResponse, …)
├── parser.py           # Response / tool-call parser
├── pipeline.py         # Post-run pipeline: evaluate → reflect → persist → store
├── reflection.py       # LLM-based structured reflection generator
├── registry.py         # Tool registry
├── run_record.py       # Execution trajectory recorder
├── tools.py            # Built-in tools (shell, read/write/edit file)
└── trajectory_store.py # Local JSON trajectory persistence
```

## Self-Improvement Loop

Every completed run goes through the **ExperiencePipeline** automatically:

1. **Evaluate** — `TrajectoryEvaluator` scores the run deterministically (completion, tool failures, verification commands).
2. **Reflect** — `ReflectionGenerator` asks an LLM to produce a structured retrospective (`what_worked`, `what_failed`, `why_it_failed`, `what_to_do_differently`).
3. **Persist** — `TrajectoryStore` saves the full trajectory JSON with embedded evaluation and reflection.
4. **Store** — `ExperienceMemory` saves the experience for future retrieval.

Before each new run, the agent retrieves the most relevant past experiences (by token-overlap + evaluation score) and injects them as optional advisory context — without overriding system instructions.

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

200 tests covering the full agent stack: trajectory recording, deterministic evaluation, LLM reflection, experience memory, pipeline orchestration, CLI integration, and all core tools.
