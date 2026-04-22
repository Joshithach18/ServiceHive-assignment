# agent.py
# Core LangGraph agent — intent classification, RAG retrieval, and lead collection.
#
# Graph topology (per turn):
#
#   START ──► [initial_router]
#                │
#       ┌────────┴──────────────┐
#       │ lead in progress      │ no active lead
#       ▼                       ▼
#  handle_lead           classify_intent
#       │                       │
#       │            ┌──────────┼──────────┐
#       │       greeting     inquiry    high_intent
#       │          ▼            ▼            │
#       │       respond    retrieve_rag      │
#       │                     ▼             │
#       │                  respond          │
#       └──────────────────────┴────────────►
#                                          END

import re
from typing import Annotated, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from rag_pipeline import RAGPipeline
from tools import mock_lead_capture

# ──────────────────────────────────────────────────────────────────────────────
# State Schema
# ──────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    messages:      Annotated[list, add_messages]  # full conversation history
    intent:        str          # greeting | inquiry | high_intent
    lead_stage:    str          # none | need_name | need_email | need_platform | captured
    lead_name:     Optional[str]
    lead_email:    Optional[str]
    lead_platform: Optional[str]
    rag_context:   str


# ──────────────────────────────────────────────────────────────────────────────
# LLM + RAG Initialisation
# ──────────────────────────────────────────────────────────────────────────────

llm = ChatGroq(
    model="llama-3.1-8b-instant",   # Free on Groq — fast Llama 3 8B
    temperature=0.3,
    max_tokens=512,
)

rag = RAGPipeline()

# ──────────────────────────────────────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """\
You are Alex, a friendly and knowledgeable AI sales assistant for AutoStream —
a SaaS platform that provides automated video editing tools for content creators.

Your goals:
1. Help users understand AutoStream's features and pricing accurately.
2. Answer questions using the knowledge base provided below.
3. Identify when users are ready to sign up and guide them toward it.
4. Be warm, concise, and professional.

Only answer questions about AutoStream. For unrelated topics, politely redirect.

{context_section}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Helper Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _extract_email(text: str) -> Optional[str]:
    """Extract a valid email address from free text."""
    match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return match.group() if match else None


def _extract_name(text: str) -> Optional[str]:
    """Use the LLM to extract a person's name from a short message."""
    prompt = (
        f'Extract only the person\'s full name from this message.\n'
        f'Return ONLY the name — no extra words, punctuation, or explanation.\n'
        f'If no name is present, return exactly: NONE\n\n'
        f'Message: "{text}"\nName:'
    )
    raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    if not raw or raw.upper() == "NONE":
        return None
    # Guard against accidental multi-word responses
    words = raw.split()
    return " ".join(words[:3])  # accept up to 3-word names


# ──────────────────────────────────────────────────────────────────────────────
# Node Functions
# ──────────────────────────────────────────────────────────────────────────────

def classify_intent(state: AgentState) -> dict:
    """
    Classify the user's latest message into one of three intents:
    greeting | inquiry | high_intent
    """
    messages = state.get("messages", [])
    last_msg  = messages[-1].content if messages else ""

    # Build short conversation snippet for context-aware classification
    recent = messages[-4:] if len(messages) >= 4 else messages
    context_lines = []
    for m in recent[:-1]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        context_lines.append(f"{role}: {m.content}")
    conv_ctx = "\n".join(context_lines)

    prefix = ""
    if conv_ctx.strip():
        prefix = "Previous conversation:\n" + conv_ctx + "\n"

    prompt = f"""\
{prefix}User's latest message: "{last_msg}"

Classify this message into EXACTLY ONE of:
- greeting   → Simple hello, thanks, or casual small talk
- inquiry    → Questions about AutoStream product, pricing, features, plans, or policies
- high_intent → User wants to sign up, buy, start a trial, or explicitly try the product

Reply with ONLY the single label word (greeting / inquiry / high_intent)."""

    raw = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()

    if "high_intent" in raw or "high intent" in raw:
        intent = "high_intent"
    elif "inquiry" in raw:
        intent = "inquiry"
    else:
        intent = "greeting"

    return {"intent": intent}


# ──────────────────────────────────────────────────────────────────────────────

def retrieve_rag_context(state: AgentState) -> dict:
    """Retrieve top-3 relevant chunks from the knowledge base."""
    messages = state.get("messages", [])
    last_msg  = messages[-1].content if messages else ""
    context   = rag.retrieve(last_msg, k=3)
    return {"rag_context": context}


# ──────────────────────────────────────────────────────────────────────────────

def generate_response(state: AgentState) -> dict:
    """Generate a response for greeting or inquiry intents."""
    rag_context = state.get("rag_context", "")

    context_section = (
        f"Relevant Knowledge Base:\n{rag_context}"
        if rag_context
        else "No specific KB context retrieved for this query."
    )

    system_msg = _SYSTEM_TEMPLATE.format(context_section=context_section)
    chain_input = [SystemMessage(content=system_msg)] + list(state.get("messages", []))
    response    = llm.invoke(chain_input)

    return {"messages": [response], "rag_context": ""}


# ──────────────────────────────────────────────────────────────────────────────

def handle_lead_collection(state: AgentState) -> dict:
    """
    Multi-step lead data collection state machine.

    Stages (stored in state["lead_stage"]):
      none          → detected high-intent; ask for name
      need_name     → waiting for name
      need_email    → waiting for email
      need_platform → waiting for platform; then fire mock_lead_capture()
      captured      → lead already saved; resume normal conversation
    """
    lead_stage = state.get("lead_stage", "none")
    messages   = state.get("messages", [])
    last_msg   = messages[-1].content if messages else ""
    updates: dict = {}

    # ── Stage: none → ask for name ───────────────────────────────────────────
    if lead_stage == "none":
        updates["lead_stage"] = "need_name"
        reply = (
            "That's fantastic! I'd love to get you set up on AutoStream Pro. 🎉\n\n"
            "To kick things off, could you share your **name**?"
        )

    # ── Stage: need_name → extract name ──────────────────────────────────────
    elif lead_stage == "need_name":
        name = _extract_name(last_msg)
        if name:
            updates["lead_name"]  = name
            updates["lead_stage"] = "need_email"
            reply = f"Great to meet you, {name}! 😊\nWhat's your **email address**?"
        else:
            reply = "I didn't quite catch your name — could you share it with me?"

    # ── Stage: need_email → extract email ────────────────────────────────────
    elif lead_stage == "need_email":
        email = _extract_email(last_msg)
        if email:
            updates["lead_email"] = email
            updates["lead_stage"] = "need_platform"
            reply = (
                "Perfect! Last question — which **platform** do you primarily "
                "create content on?\n(e.g., YouTube, Instagram, TikTok, Twitch …)"
            )
        else:
            reply = (
                "I couldn't spot a valid email in that. "
                "Could you please share your email address?"
            )

    # ── Stage: need_platform → capture lead ──────────────────────────────────
    elif lead_stage == "need_platform":
        platform = last_msg.strip()
        if platform:
            lead_name  = state.get("lead_name", "")
            lead_email = state.get("lead_email", "")

            updates["lead_platform"] = platform
            updates["lead_stage"]    = "captured"

            # 🔧 Tool Execution — fires only when all three fields are collected
            mock_lead_capture(lead_name, lead_email, platform)

            reply = (
                f"🎉 **You're all set, {lead_name}!**\n\n"
                f"Here's what we've captured:\n"
                f"- **Name:** {lead_name}\n"
                f"- **Email:** {lead_email}\n"
                f"- **Platform:** {platform}\n\n"
                f"Our team will reach out at **{lead_email}** within 24 hours "
                f"to help you get started with AutoStream Pro. Welcome aboard! 🚀\n\n"
                f"Is there anything else I can help you with?"
            )
        else:
            reply = "Could you let me know which platform you mainly create on?"

    # ── Stage: captured → normal conversation ────────────────────────────────
    else:
        system_msg  = _SYSTEM_TEMPLATE.format(context_section="")
        chain_input = [SystemMessage(content=system_msg)] + list(messages)
        reply       = llm.invoke(chain_input).content

    updates["messages"] = [AIMessage(content=reply)]
    return updates


# ──────────────────────────────────────────────────────────────────────────────
# Routing Logic
# ──────────────────────────────────────────────────────────────────────────────

def initial_router(state: AgentState) -> str:
    """
    If a lead collection is already in progress, skip intent classification
    and go straight to the lead handler.
    """
    lead_stage = state.get("lead_stage", "none")
    if lead_stage not in ("none", "captured", None):
        return "handle_lead"
    return "classify_intent"


def intent_router(state: AgentState) -> str:
    """Route to the correct sub-graph based on classified intent."""
    intent     = state.get("intent", "inquiry")
    lead_stage = state.get("lead_stage", "none")

    if intent == "high_intent" and lead_stage != "captured":
        return "handle_lead"
    elif intent == "inquiry":
        return "retrieve_rag"
    else:
        return "respond"


# ──────────────────────────────────────────────────────────────────────────────
# Build + Compile the LangGraph
# ──────────────────────────────────────────────────────────────────────────────

def build_agent():
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("classify_intent",    classify_intent)
    workflow.add_node("retrieve_rag",       retrieve_rag_context)
    workflow.add_node("respond",            generate_response)
    workflow.add_node("handle_lead",        handle_lead_collection)

    # Entry-point conditional edge
    workflow.add_conditional_edges(
        START,
        initial_router,
        {"handle_lead": "handle_lead", "classify_intent": "classify_intent"},
    )

    # Post-intent-classification routing
    workflow.add_conditional_edges(
        "classify_intent",
        intent_router,
        {
            "handle_lead": "handle_lead",
            "retrieve_rag": "retrieve_rag",
            "respond": "respond",
        },
    )

    # RAG → generate response
    workflow.add_edge("retrieve_rag", "respond")

    # Terminal edges
    workflow.add_edge("respond",     END)
    workflow.add_edge("handle_lead", END)

    # Compile with in-memory checkpointer (persists state across turns)
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# Singleton — imported by main.py
agent_app = build_agent()