# Terminal Coding Agent

A clean, modular terminal-based coding agent built from scratch.

## Project Structure

```
.
├── pyproject.toml
├── README.md
├── src/
│   └── coding_agent/
│       ├── __init__.py
│       ├── __main__.py
│       └── cli.py
└── tests/
    ├── __init__.py
    └── test_cli.py
```

## Running the Application

Run directly using Python:

```bash
python -m coding_agent
```

Or install in editable mode:

```bash
pip install -e .
coding-agent
```

## Running Tests

Run test suite using Python's standard `unittest`:

```bash
python -m unittest discover -s tests
```
