from langgraph.graph import MessagesState


class State(MessagesState):
    answer: str = ""
    client_id: str = ""
