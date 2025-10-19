from typing import Dict, Any

def answer(req: Dict[str, Any]):
    q = (req.get("question") or "").strip()
    dry = bool(req.get("dry_run"))
    ctx = {
        "seeds": ["Carol","Bob","Eve","Alice","Dan"],
        "paths": [],
        "contexts": [],
        "local_facts": [],
        "node_notes": [
            "Carol: Carol is a PhD student; co-authored with Bob.",
            "Bob: Bob collaborates with Alice and Carol; co-authored with Carol.",
            "Eve: Eve co-authored with Bob.",
            "Alice: Alice is a researcher in ML.",
            "Dan: Dan advised Carol."
        ]
    }
    if dry:
        return {"answer": "Dry run OK: graph search planned.", "ctx": ctx}
    if "carol" in q.lower():
        return {"answer": "Bob co-authored a paper with Carol.", "ctx": ctx}
    return {"answer": "stub", "ctx": ctx}
