# Customer Support Ticket Triage System

An agentic customer support ticket triage desk built with **LangGraph**, **FastAPI**, and **LangChain Google GenAI**. The system triages incoming support tickets by category and urgency, reviews draft responses for quality and completeness, integrates order lookup tools for billing tickets, pauses for human approval on billing issues via LangGraph interrupts, and persists conversation history across customer sessions using SQLite checkpointing.

---

## 1. Run Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.11 / 3.14)
- A Google Gemini API key

### Installation
Clone or navigate to the project directory and install the required dependencies:

```bash
# Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install fastapi uvicorn python-dotenv langchain-core langchain-google-genai langgraph langgraph-checkpoint langgraph-checkpoint-sqlite
```

### Environment Configuration (.env)
Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
# Optional checkpointer toggle: "sqlite" (default) or "memory"
CHECKPOINTER=sqlite
```

### Running the Application
Launch the FastAPI development server with auto-reload:

```bash
uvicorn app:app --reload
```

Once running, navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## 2. LLM Used
- **Gemini 2.5 Flash** (via `langchain-google-genai`), with fallback to `gemini-3.5-flash` / `gemini-3.8-flash` if model deprecation or quota switches apply.
- Configured with `temperature=0.2` and structured outputs for classification and strict reviewer quality checks.

---

## 3. Persistence
- **Configurable Checkpointer**: Controlled via the `CHECKPOINTER` environment variable.
  - `CHECKPOINTER=memory`: Uses LangGraph's `InMemorySaver` (ephemeral).
  - Default (`CHECKPOINTER=sqlite`): Uses `SqliteSaver` connected to `tickets.db`.
- **Customer Thread Continuity**: Each Customer ID acts as the LangGraph `thread_id`. Reusing the same Customer ID retrieves past messages, drafts, and state from the database, allowing returning customers to resume conversations seamlessly.

---

## 4. Something Tricky: Key Architectural Insights

1. **State Persistence Across Subsequent Tickets**:
   Because conversation history persists on the thread across multiple ticket interactions, per-ticket tracking attributes such as `attempts`, `score`, `feedback`, and `draft` must be explicitly reset in the `classify` node. If not cleared, a returning customer submitting a second ticket would retain previous reviewer scores or attempts from earlier tickets, causing premature routing or routing loops.

2. **LangGraph `interrupt()` Resumption & Node Idempotency**:
   When `Command(resume=...)` is invoked to resume the graph after an interrupt, LangGraph resumes and re-executes the interrupted node (`human_approval`) with the provided resumption payload. Therefore, nodes containing `interrupt()` calls must have no pre-interrupt side effects (such as external API calls or database writes) to ensure safe, idempotent resumption.

---

## 5. Testing Checklist

Use the following tests to verify all core flows in the application:

- [ ] **Billing Ticket with Approval Flow**:
  1. Open [http://127.0.0.1:8000](http://127.0.0.1:8000).
  2. Enter Customer ID `customer-001`.
  3. Enter message: `"I was charged twice for order #1042"` and click **Send ticket**.
  4. Verify the ticket is classified as `billing`, triage details are displayed (category, urgency, review score, attempts, draft reply), and the **Billing Approval Required** panel appears with the draft pre-filled in the editable textarea.
  5. Verify that the draft quotes the two duplicate payments of `$89.99` on November 1 retrieved from the order tool.
  6. Click **Approve & send** (or edit and click **Save edit & send**).
  7. Confirm the approval panel closes and the final reply appears in the **Conversation Thread**.

- [ ] **Technical Ticket Automatic Send**:
  1. Enter Customer ID `tech-user-123`.
  2. Enter message: `"The app crashes on settings"` and click **Send ticket**.
  3. Verify the ticket is classified as `technical`.
  4. Confirm that no approval panel appears and the reply is automatically sent and displayed in the thread.

- [ ] **SQLite Persistence & Returning Customers**:
  1. Submit tickets with a Customer ID (e.g. `customer-001`).
  2. Restart the server (`CTRL+C` then `uvicorn app:app --reload`).
  3. Reload the webpage and type `customer-001` into the Customer ID box.
  4. Verify that the checkpointer fetches and displays the complete previous message history from `tickets.db`.
