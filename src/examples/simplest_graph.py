
from typing import List

from packages.raphlib.src.raphlib import ChatHistory
from packages.raphlib.src.raphlib.graph import GraphBuilder, BaseState

class State(BaseState):
    history: ChatHistory
    
G = GraphBuilder[State](start="node", state=State)

@G.node(next="__end__")
async def node(s: State):
    for l in "hello":
        yield l
    s.history.append("ai", "hello")

graph = G.compiled()
