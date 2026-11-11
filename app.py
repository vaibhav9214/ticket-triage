import os
from typing import Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langgraph.types import Command
from agent import graph

app = FastAPI(title="Support Desk")

def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content) if content is not None else ""

class TicketRequest(BaseModel):
    customer_id: str
    message: str

class ApproveRequest(BaseModel):
    customer_id: str
    action: str
    text: Optional[str] = None

MAX_ID_LEN = 200
MAX_MSG_LEN = 10000

@app.post("/api/ticket")
def submit_ticket(req: TicketRequest):
    cid = req.customer_id.strip()
    msg = req.message.strip()
    print(f"--> [submit_ticket] Received for {cid}: {msg[:40]}...", flush=True)
    if not cid:
        raise HTTPException(status_code=400, detail="Customer ID is required")
    if not msg:
        raise HTTPException(status_code=400, detail="Message is required")
    if len(cid) > MAX_ID_LEN:
        raise HTTPException(status_code=400, detail="Customer ID is too long")
    if len(msg) > MAX_MSG_LEN:
        raise HTTPException(status_code=400, detail="Message is too long")

    cfg = {"configurable": {"thread_id": cid}}
    print(f"--> [submit_ticket] Invoking graph for {cid}...", flush=True)
    try:
      out = graph.invoke(
        {"messages": [("user", msg)]},
        cfg
    )

    except Exception:
      import traceback
      traceback.print_exc()

      raise HTTPException(
        status_code=500,
        detail="Ticket processing failed. Check backend terminal."
    )
    print(f"--> [submit_ticket] Graph finished for {cid}! Out keys: {list(out.keys())}", flush=True)
    state = graph.get_state(cfg)
    values = state.values or {}

    is_interrupted = bool(out.get("__interrupt__") or ("human_approval" in (state.next or ())))
    status = "pending_approval" if is_interrupted else "sent"
    print(f"--> [submit_ticket] Completed with status={status}", flush=True)

    return {
        "status": status,
        "category": values.get("category", ""),
        "urgency": values.get("urgency", ""),
        "score": values.get("score", 0),
        "attempts": values.get("attempts", 0),
        "draft": extract_text(values.get("draft", "")),
        "final_reply": extract_text(out.get("final_reply") or values.get("final_reply", ""))
    }

@app.post("/api/approve")
def approve_ticket(req: ApproveRequest):
    cid = req.customer_id.strip()
    print(f"--> [approve_ticket] Received for {cid}, action={req.action}", flush=True)
    if not cid:
        raise HTTPException(status_code=400, detail="Customer ID is required")
    if req.action not in ("approve", "edit"):
        raise HTTPException(status_code=400, detail="Action must be 'approve' or 'edit'")

    cfg = {"configurable": {"thread_id": cid}}
    state = graph.get_state(cfg)
    if not state.next or "human_approval" not in state.next:
        raise HTTPException(status_code=400, detail="No pending approval found for this customer thread")

    resume_payload = {"action": req.action}

    if req.action == "edit":
        if not req.text or not req.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Edited reply cannot be empty"
            )
        if len(req.text) > MAX_MSG_LEN:
            raise HTTPException(
                status_code=400,
                detail="Edited reply is too long"
            )
        resume_payload["text"] = req.text.strip()

    print(f"--> [approve_ticket] Resuming graph for {cid}...", flush=True)
    out = graph.invoke(Command(resume=resume_payload), cfg)
    print(f"--> [approve_ticket] Resume graph finished for {cid}", flush=True)
    state = graph.get_state(cfg)
    values = state.values or {}

    return {
        "status": "sent",
        "category": values.get("category", ""),
        "urgency": values.get("urgency", ""),
        "score": values.get("score", 0),
        "attempts": values.get("attempts", 0),
        "draft": extract_text(values.get("draft", "")),
        "final_reply": extract_text(out.get("final_reply") or values.get("final_reply", ""))
    }

@app.get("/api/thread/{id}")
def get_thread(id: str):
    cid = id.strip()
    cfg = {"configurable": {"thread_id": cid}}
    state = graph.get_state(cfg)
    raw_messages = (state.values or {}).get("messages", [])
    history = []
    for m in raw_messages:
        role = "customer" if m.type == "human" else "agent"
        history.append({
            "role": role,
            "content": extract_text(m.content)
        })
    return {
        "customer_id": cid,
        "messages": history
    }

# Serve single static page and static assets
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def get_index():
    return FileResponse("static/index.html")
