from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage
from src.agents.state import State
from langgraph.checkpoint.memory import MemorySaver
from src.agents.chat.chat import ChatAgent

VALID_AGENTS = ["chat"]


def start(state: State) -> State:
    return state


chat = ChatAgent()
workflow = StateGraph(State)


workflow.add_node("start", start)
workflow.add_node("chat", chat.process)
workflow.set_entry_point("start")
workflow.add_edge("start", "chat")

graph = workflow.compile(checkpointer=MemorySaver())
