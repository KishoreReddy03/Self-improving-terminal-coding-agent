"""CLI entry point for the terminal coding agent."""

import argparse
import sys
from typing import Any, Dict, List, Optional

from coding_agent.agent import Agent
from coding_agent.config import load_config_from_env
from coding_agent.memory import ExperienceMemory
from coding_agent.models import AgentResult
from coding_agent.pipeline import ExperiencePipeline
from coding_agent.trajectory_store import TrajectoryStore


PLAN_MODE_DIRECTIVE = (
    "\n\n[PLAN MODE INSTRUCTION]: Please first analyze the request and outline a clear, "
    "step-by-step implementation plan before proceeding with execution."
)


def _run_pipeline(
    result: AgentResult,
    pipeline: Optional[ExperiencePipeline],
) -> None:
    """Run the post-run experience pipeline safely without crashing the CLI."""
    if pipeline is None or result.run_record is None:
        return
    try:
        pipeline.process(result)
    except Exception as err:
        print(f"\nWarning: Post-run pipeline failed: {err}", file=sys.stderr)


def run_interactive_session(
    agent: Agent,
    start_in_plan_mode: bool = False,
    pipeline: Optional[ExperiencePipeline] = None,
) -> int:
    """Run interactive REPL terminal loop for the agent."""
    conversation: List[Dict[str, Any]] = []
    plan_mode: bool = start_in_plan_mode

    print("Terminal Coding Agent v0.1.0")
    print("Type /plan to toggle plan mode, /clear to reset history, or /exit to quit.\n")

    if plan_mode:
        print("Plan mode is currently ENABLED.\n")

    while True:
        prompt_prefix = "[PLAN] agent> " if plan_mode else "agent> "
        try:
            user_input = input(prompt_prefix).strip()
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt (type /exit or press Ctrl-D to quit)")
            continue
        except EOFError:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            print("Goodbye!")
            break

        if user_input.lower() == "/plan":
            plan_mode = not plan_mode
            status = "ENABLED" if plan_mode else "DISABLED"
            print(f"Plan mode is now {status}.\n")
            continue

        if user_input.lower() == "/clear":
            conversation = []
            print("Conversation history cleared.\n")
            continue

        if user_input.lower() == "/help":
            print("Available commands:")
            print("  /plan  - Toggle plan mode on/off")
            print("  /clear - Clear current conversation history")
            print("  /exit  - Quit the application\n")
            continue

        message_content = user_input
        if plan_mode:
            message_content += PLAN_MODE_DIRECTIVE

        conversation.append({"role": "user", "content": message_content})

        try:
            result = agent.run(conversation, plan_mode=plan_mode)
            conversation = result.messages
            _run_pipeline(result, pipeline)
            if result.final_response:
                print(f"\n{result.final_response}\n")
        except Exception as e:
            print(f"\nError during agent execution: {e}\n")

    return 0


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI execution entry point."""
    parser = argparse.ArgumentParser(description="Terminal Coding Agent CLI")
    parser.add_argument("prompt", nargs="?", help="Optional single prompt to execute non-interactively")
    parser.add_argument("--plan", action="store_true", help="Start in plan mode")
    parser.add_argument("--auto-approve", action="store_true", help="Auto approve write-capable tools without prompting")
    parser.add_argument("--env-file", help="Path to custom .env file")
    parser.add_argument("--trajectory-dir", help="Custom directory for trajectory JSON storage")
    parser.add_argument("--memory-dir", help="Custom directory for experience memory storage")
    parser.add_argument("--no-memory", action="store_true", help="Disable experience memory (no retrieval or storage)")
    parser.add_argument("--no-reflection", action="store_true", help="Disable LLM reflection generation after each run")
    parser.add_argument("--only-store-successful", action="store_true", help="Only store experiences from successful runs")
    parser.add_argument("--verbose-pipeline", action="store_true", help="Print pipeline status messages to stderr")

    parsed_args = parser.parse_args(args)

    config = load_config_from_env(env_path=parsed_args.env_file if parsed_args.env_file else None)

    # Set up memory and pipeline
    memory: Optional[ExperienceMemory] = None
    pipeline: Optional[ExperiencePipeline] = None

    if not parsed_args.no_memory:
        memory = ExperienceMemory(storage_dir=parsed_args.memory_dir)
        trajectory_store = TrajectoryStore(storage_dir=parsed_args.trajectory_dir)
        pipeline = ExperiencePipeline(
            trajectory_store=trajectory_store,
            memory=memory,
            only_store_successful=parsed_args.only_store_successful,
            enable_reflection=not parsed_args.no_reflection,
            verbose=parsed_args.verbose_pipeline,
        )
    else:
        # Memory disabled — still persist trajectories if a dir was specified
        trajectory_store = TrajectoryStore(storage_dir=parsed_args.trajectory_dir)
        pipeline = ExperiencePipeline(
            trajectory_store=trajectory_store,
            memory=None,
            enable_reflection=not parsed_args.no_reflection,
            verbose=parsed_args.verbose_pipeline,
        )

    agent = Agent(
        config=config,
        auto_approve=parsed_args.auto_approve,
        plan_mode=parsed_args.plan,
        memory=memory,
    )

    print("Terminal Coding Agent initialized successfully.")
    print(f"Model Provider URL: {config.provider_url}")
    print(f"Model Name:         {config.model_name}")
    if memory is not None:
        memory_dir = memory.storage_dir
        print(f"Experience Memory:  {memory_dir}")

    if parsed_args.prompt:
        user_content = parsed_args.prompt
        if parsed_args.plan:
            user_content += PLAN_MODE_DIRECTIVE
        result = agent.run(user_content, plan_mode=parsed_args.plan)
        _run_pipeline(result, pipeline)
        if result.final_response:
            print(f"\n{result.final_response}")
        return 0

    return run_interactive_session(
        agent, start_in_plan_mode=parsed_args.plan, pipeline=pipeline
    )


if __name__ == "__main__":
    sys.exit(main())
