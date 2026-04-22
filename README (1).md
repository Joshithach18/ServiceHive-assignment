# AutoStream AI Agent — Social-to-Lead Agentic Workflow

> Built for the **ServiceHive × Inflx** Machine Learning Intern Assignment  
> A production-grade GenAI conversational agent with Intent Detection, RAG, and Lead Capture

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Setup & Installation](#setup--installation)
6. [How to Run Locally](#how-to-run-locally)
7. [Expected Conversation Flow](#expected-conversation-flow)
8. [WhatsApp Deployment via Webhooks](#whatsapp-deployment-via-webhooks)

---

## Project Overview

This agent acts as **Alex**, an AI sales assistant for **AutoStream** — a fictional SaaS product providing automated video editing tools for content creators. The agent can:

- Greet users and handle casual conversation
- Answer product, pricing, and policy questions via **RAG** (Retrieval-Augmented Generation)
- Detect **high-intent** users ready to sign up
- Collect name, email, and creator platform through a **multi-turn state machine**
- Fire a **lead capture tool** only after all three fields are confirmed

---

## Architecture

### Why LangGraph?

LangGraph was chosen over plain LangChain or AutoGen for three reasons:

1. **Explicit state management** — LangGraph's `TypedDict`-based state persists every field (intent, lead stage, collected info, conversation history) across all turns automatically via `MemorySaver`. There is no risk of state being lost mid-collection.

2. **Conditional branching** — The agent's behaviour fundamentally branches per turn (greeting → casual reply; inquiry → RAG → reply; high-intent → 4-step collection). LangGraph's `add_conditional_edges` makes this control flow first-class rather than embedded in prompt logic.

3. **Tool invocation safety** — Because the lead-capture tool lives inside a dedicated `handle_lead` node that is only reached after all three fields are validated, there is zero risk of premature tool firing.

### How State Is Managed

```
AgentState (persisted via MemorySaver across every turn)
├── messages       — full conversation history (Annotated with add_messages reducer)
├── intent         — greeting | inquiry | high_intent  (updated each turn)
├── lead_stage     — none → need_name → need_email → need_platform → captured
├── lead_name      — collected in need_name stage
├── lead_email     — collected in need_email stage (regex-validated)
├── lead_platform  — collected in need_platform stage
└── rag_context    — top-3 KB chunks for current turn (cleared after use)
```

Each turn, the graph is invoked with only the new `HumanMessage`. The `MemorySaver` checkpointer replays and updates the full state transparently.

### Graph Topology

```
START
  │
  ▼  [initial_router]
  ├─── lead in progress ──────────────► handle_lead ──► END
  │
  └─── no active lead ─────────────► classify_intent
                                           │
                       ┌───────────────────┼───────────────────┐
                    greeting            inquiry           high_intent
                       │                  │                    │
                       ▼                  ▼                    ▼
                    respond          retrieve_rag         handle_lead
                       │                  │                    │
                      END              respond                END
                                          │
                                         END
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Agent Framework | LangGraph 0.2+ |
| LLM | Claude 3 Haiku (`claude-3-haiku-20240307`) |
| LLM Client | LangChain-Anthropic |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, CPU) |
| Vector Store | FAISS (in-memory) |
| State Persistence | `MemorySaver` (LangGraph built-in) |
| Knowledge Base | Markdown file (local RAG) |

---

## Project Structure

```
autostream-agent/
├── knowledge_base/
│   └── autostream_kb.md       # AutoStream pricing, features, and policies
│
├── agent.py                   # LangGraph graph — nodes, routing, state schema
├── rag_pipeline.py            # FAISS + sentence-transformers RAG class
├── tools.py                   # mock_lead_capture() tool
├── main.py                    # CLI entry point
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.9 or higher
- An [Anthropic API key](https://console.anthropic.com/) (free tier works)

### Step 1 — Clone the repository
```bash
git clone https://github.com/<your-username>/autostream-agent.git
cd autostream-agent
```

### Step 2 — Create a virtual environment
```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ The first run will download the `all-MiniLM-L6-v2` embedding model (~80 MB). This is a one-time download cached locally.

### Step 4 — Configure your API key
```bash
cp .env.example .env
```
Open `.env` and replace the placeholder with your actual key:
```
ANTHROPIC_API_KEY=sk-ant-your-real-key-here
```

---

## How to Run Locally

```bash
python main.py
```

You will see the welcome banner and can start chatting:

```
╔═══════════════════════════════════════════════════════╗
║      AutoStream AI Assistant  —  Powered by Inflx     ║
╚═══════════════════════════════════════════════════════╝

Hi! I'm Alex, your AutoStream assistant. 👋

You: Hi, tell me about your pricing.

Alex: AutoStream offers two plans:
  - Basic ($29/month): 10 videos/month, 720p resolution
  - Pro ($79/month): Unlimited videos, 4K, AI captions, 24/7 support
...
```

Type `quit` or `exit` to end the session.

---

## Expected Conversation Flow

```
User  → "Hi, tell me about your pricing."
Alex  → [RAG] Retrieves pricing chunks → explains Basic & Pro plans

User  → "That sounds great. I want to try the Pro plan for my YouTube channel."
Alex  → [Intent: high_intent] "Fantastic! Could you share your name?"

User  → "My name is Sarah"
Alex  → "Great to meet you, Sarah! What's your email address?"

User  → "sarah@example.com"
Alex  → "Perfect! Which platform do you create content on?"

User  → "YouTube"
Alex  → [mock_lead_capture fired] "You're all set, Sarah! ..."

Console output:
  ✅  LEAD CAPTURED SUCCESSFULLY
     Name     : Sarah
     Email    : sarah@example.com
     Platform : YouTube
```

---

## WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, you would use the **WhatsApp Business Cloud API** (Meta) combined with a webhook server:

### Architecture

```
WhatsApp User
     │  (sends message)
     ▼
Meta WhatsApp Cloud API
     │  POST /webhook  (JSON payload with message text + sender ID)
     ▼
Your Webhook Server  (FastAPI / Flask)
     │  extract text, map sender_id → thread_id
     ▼
agent_app.invoke({"messages": [HumanMessage(...)]}, config={"thread_id": sender_id})
     │  AI response string
     ▼
Meta Send Message API  (POST with reply text)
     │
     ▼
WhatsApp User  (receives reply)
```

### Implementation Steps

1. **Create a Meta Developer App** and enable the WhatsApp Business product.

2. **Build a webhook endpoint** (e.g., with FastAPI):
   ```python
   from fastapi import FastAPI, Request
   from langchain_core.messages import HumanMessage
   from agent import agent_app
   import httpx

   app = FastAPI()

   @app.post("/webhook")
   async def whatsapp_webhook(request: Request):
       data      = await request.json()
       message   = data["entry"][0]["changes"][0]["value"]["messages"][0]
       sender_id = message["from"]      # unique per user → use as thread_id
       text      = message["text"]["body"]

       config   = {"configurable": {"thread_id": sender_id}}
       result   = agent_app.invoke({"messages": [HumanMessage(content=text)]}, config=config)
       reply    = result["messages"][-1].content

       # Send reply back via WhatsApp Cloud API
       async with httpx.AsyncClient() as client:
           await client.post(
               f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages",
               headers={"Authorization": f"Bearer {WA_TOKEN}"},
               json={"messaging_product": "whatsapp", "to": sender_id,
                     "type": "text", "text": {"body": reply}},
           )
       return {"status": "ok"}
   ```

3. **Register your webhook URL** in the Meta Developer Console and verify it with a `GET /webhook` challenge endpoint.

4. **Host on a public server** (Railway, Render, or any cloud VM) — WhatsApp requires HTTPS.

5. **Session isolation** is already handled: each `sender_id` maps to its own `thread_id`, so the `MemorySaver` maintains separate lead-collection state per user automatically.

---

## Evaluation Checklist

| Criterion | Implementation |
|---|---|
| Intent detection | `classify_intent` node using LLM with conversation context |
| RAG pipeline | FAISS + sentence-transformers + `autostream_kb.md` |
| State management | LangGraph `MemorySaver` across 5–6 turns |
| Tool calling safety | `mock_lead_capture` fires only after all 3 fields confirmed |
| Code clarity | Modular files, type hints, inline comments |
| Deployability | Webhook design documented; stateless server-ready |
