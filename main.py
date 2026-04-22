# main.py
# CLI entry point for the AutoStream AI Agent.
# Run:  python main.py

import os
import sys

from dotenv import load_dotenv

# Load .env before importing agent (which initialises the LLM)
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    sys.exit(
        "\n❌  GROQ_API_KEY is not set.\n"
        "    1. Get your FREE key at: https://console.groq.com/keys\n"
        "       (No credit card — takes 30 seconds)\n"
        "    2. Add it to a .env file in the project root:\n"
        "           GROQ_API_KEY=gsk_...\n"
        "    See .env.example for reference.\n"
    )

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from agent import agent_app  # noqa: E402 — import after env check

# ──────────────────────────────────────────────────────────────────────────────
# Session config — single thread keeps state across all turns
# ──────────────────────────────────────────────────────────────────────────────
SESSION_CONFIG = {"configurable": {"thread_id": "autostream-demo-session"}}

WELCOME = """
╔═══════════════════════════════════════════════════════╗
║      AutoStream AI Assistant  —  Powered by Inflx     ║
║      Type  'quit'  or  'exit'  to end the session     ║
╚═══════════════════════════════════════════════════════╝

Hi! I'm Alex, your AutoStream assistant. 👋
I can tell you about our video editing plans, pricing, and help you get started!

What can I help you with today?
"""


# ──────────────────────────────────────────────────────────────────────────────

def chat(user_input: str) -> str:
    """Send one user message to the agent and return the AI reply."""
    result   = agent_app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=SESSION_CONFIG,
    )
    messages = result.get("messages", [])
    # Return the last AI message in the thread
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content
    return "Sorry, I couldn't generate a response."


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(WELCOME)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! Thanks for chatting with AutoStream. 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "bye", "goodbye"}:
            print("\nGoodbye! Thanks for chatting with AutoStream. 👋")
            break

        print("\nAlex: ", end="", flush=True)
        response = chat(user_input)
        print(response)
        print()


if __name__ == "__main__":
    main()