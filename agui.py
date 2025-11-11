from uuid import uuid4
from datetime import datetime
from typing import Dict, Any, List, Generator


def build_event(event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    e: Dict[str, Any] = {"type": event_type}
    e.update(payload)
    return e


def demo_events() -> List[Dict[str, Any]]:
    run_id = str(uuid4())
    now = datetime.utcnow().isoformat()
    return [
        build_event("run_start", {"run_id": run_id, "timestamp": now}),
        build_event("message", {"role": "user", "content": "Parse uploaded statements and summarize totals."}),
        build_event("tool_call", {"tool": "docling.convert", "args": {"files": "session.uploads"}}),
        build_event("progress", {"content": "Converting documents and extracting tables...", "progress": 0.35}),
        build_event("tool_result", {"tool": "docling.convert", "result": {"tables_found": True}}),
        build_event("progress", {"content": "Normalizing transactions and computing totals...", "progress": 0.75}),
        build_event("message", {"role": "assistant", "content": "Found 3 documents, 524 transactions; total credits €12,453.90, debits €11,012.45."}),
        build_event("run_end", {"timestamp": datetime.utcnow().isoformat(), "status": "success"}),
    ]


def stream_demo_events() -> Generator[Dict[str, Any], None, None]:
    for e in demo_events():
        yield e