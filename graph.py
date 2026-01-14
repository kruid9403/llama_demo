# graph.py
from langgraph.graph import StateGraph
from state import ResearchState
from nodes import retrieval_node, rerank_node, generation_node

graph = StateGraph(ResearchState)

graph.add_node("retrieve", retrieval_node)
graph.add_node("rerank", rerank_node)
graph.add_node("generate", generation_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "rerank")
graph.add_edge("rerank", "generate")
graph.set_finish_point("generate")

app = graph.compile()
