import os, re, sqlite3
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()
_model = os.getenv("MODEL_NAME", "gemini-3.1-flash-lite")
llm = ChatGoogleGenerativeAI(model=_model, temperature=0.2)
MAX_ATTEMPTS, PASS_SCORE = 3, 7

# ---------- Mock order data + tool ----------
ORDERS = {
    "1042": {"item": "Wireless Headphones", "total": 89.99, "status": "Delivered",
             "payments": [{"id": "pay_1", "amount": 89.99, "date": "2026-11-01"},
                          {"id": "pay_2", "amount": 89.99, "date": "2026-11-01"}]},
    "1043": {"item": "Phone Case", "total": 19.50, "status": "Shipped",
             "payments": [{"id": "pay_3", "amount": 19.50, "date": "2026-11-02"}]},
}

@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order by its numeric ID and return items, status and payments."""
    return ORDERS.get(order_id, {"error": f"Order {order_id} not found"})

# ---------- State ----------
class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    category: str
    urgency: str
    order_info: dict
    draft: str
    score: int
    feedback: str
    attempts: int
    final_reply: str

class Classification(BaseModel):
    category: Literal["billing", "technical", "general"]
    urgency: Literal["low", "medium", "high"]

class Review(BaseModel):
    score: int = Field(ge=1, le=10)
    feedback: str

# ---------- Nodes ----------
def classify(state: State):
    result = llm.with_structured_output(Classification).invoke(
        [SystemMessage(content="Classify the latest support message by category and urgency. "
                               "Use the earlier conversation for context.")] + state["messages"][-6:])
    # reset per-ticket fields; messages persist on the thread
    return {"category": result.category, "urgency": result.urgency,
            "attempts": 0, "score": 0, "feedback": "", "draft": "", "order_info": {}}

def make_handler(category: str, role: str):
    def handler(state: State):
        extra, update = "", {}
        if category == "billing":
            latest_message = next(
                m for m in reversed(state["messages"])
            
            if m.type == "human")
            text = latest_message.content
            ids = re.findall(r"#?(\d{4,})", text)
            if ids:
                info = lookup_order.invoke({"order_id": ids[-1]})
                update["order_info"] = info
                extra = f"\nOrder data (from tool): {info}"
        if state.get("feedback"):
            extra += f"\nReviewer feedback to address: {state['feedback']}"
        prompt = SystemMessage(content=f"{role} Urgency: {state['urgency']}. "
                                       f"Write a polite, concise reply to the customer.{extra}")
        reply = llm.invoke([prompt] + state["messages"][-8:])
        content = reply.content
        if isinstance(content, list):
            content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        return {**update, "draft": content, "attempts": state.get("attempts", 0) + 1}
    return handler

def reviewer(state: State):
    r = llm.with_structured_output(Review).invoke([
        SystemMessage(content="You are a strict support QA reviewer. Score 1-10 for accuracy, "
                              "empathy, clarity and completeness. Give actionable feedback."),
        HumanMessage(content=f"Customer conversation: {[m.content for m in state['messages'][-6:]]}\n"
                             f"Order data: {state.get('order_info')}\nDraft reply:\n{state['draft']}")])
    return {"score": r.score, "feedback": r.feedback}

def human_approval(state: State):
    decision = interrupt({
        "draft": state["draft"],
        "category": state["category"],
        "urgency": state["urgency"]
    })

    if decision.get("action") == "edit":
        edited_text = decision.get("text", "").strip()

        if not edited_text:
            raise ValueError("Edited reply cannot be empty")

        return {"draft": edited_text}

    return {}

def send(state: State):
    return {"final_reply": state["draft"], "messages": [AIMessage(content=state["draft"])]}

# ---------- Routing ----------
def route_category(state: State):
    return f"{state['category']}_handler"

def route_review(state: State):
    if state["score"] >= PASS_SCORE or state["attempts"] >= MAX_ATTEMPTS:
        return "human_approval" if state["category"] == "billing" else "send"
    return f"{state['category']}_handler"

# ---------- Build graph ----------
def build_graph(checkpointer):
    g = StateGraph(State)
    g.add_node("classify", classify)
    g.add_node("billing_handler", make_handler("billing", "You are a billing support specialist."))
    g.add_node("technical_handler", make_handler("technical", "You are a technical support engineer."))
    g.add_node("general_handler", make_handler("general", "You are a friendly customer support agent."))
    g.add_node("reviewer", reviewer)
    g.add_node("human_approval", human_approval)
    g.add_node("send", send)

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route_category,
        ["billing_handler", "technical_handler", "general_handler"])
    for h in ["billing_handler", "technical_handler", "general_handler"]:
        g.add_edge(h, "reviewer")
    g.add_conditional_edges("reviewer", route_review,
        ["billing_handler", "technical_handler", "general_handler", "human_approval", "send"])
    g.add_edge("human_approval", "send")
    g.add_edge("send", END)
    return g.compile(checkpointer=checkpointer)

# Switch with CHECKPOINTER=memory or sqlite (default)
def get_checkpointer():
    if os.getenv("CHECKPOINTER", "sqlite") == "memory":
        return InMemorySaver()

    conn = sqlite3.connect(
        "tickets.db",
        check_same_thread=False
    )

    checkpointer = SqliteSaver(conn)
    checkpointer.setup()

    return checkpointer

graph = build_graph(get_checkpointer())

if __name__ == "__main__":
    # Render graph image (draw_mermaid_png needs internet; otherwise save the mermaid text)
    try:
        open("graph.png", "wb").write(graph.get_graph().draw_mermaid_png())
    except Exception:
        open("graph.mmd", "w").write(graph.get_graph().draw_mermaid())

    cfg = {"configurable": {"thread_id": "customer-001"}}
    out = graph.invoke({"messages": [("user", "I was charged twice for order #1042")]}, cfg)
    if "__interrupt__" in out:
        print("DRAFT FOR APPROVAL:\n", out["__interrupt__"][0].value["draft"])
        out = graph.invoke(Command(resume={"action": "approve"}), cfg)
    print("FINAL:", out["final_reply"])