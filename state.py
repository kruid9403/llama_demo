# state.py
from typing import TypedDict, List, Dict, Any

class ResearchState(TypedDict, total=False):
    question: str
    retrieved: List[Dict]
    answer: str
    history: List[Dict[str, str]]
    stream_queue: Any
    url: str
