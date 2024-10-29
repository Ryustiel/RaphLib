import logging

from typing import List
from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI

from raphlib import LLMFunction, ChatHistory, LLMWithTools, LLMFunctionResult, setup_env

# ================================================================= TOOLS =================================================================



# ================================================================= LLMS ==================================================================

setup_env()

LLM = AzureChatOpenAI(
        deployment_name="gpt-4o",
        temperature=0.2,
        max_tokens=500,
    )

# ================================================================= STATE =================================================================

class State(BaseModel):
    CHAT: ChatHistory
    next: str
    out: List[str]

DEFAULT_STATE = State(
    CHAT = ChatHistory(types={
        "ai": "AIMessage", 
        "system": "SystemMessage", 
        "human": "HumanMessage", 
        "temp": "SystemMessage", 
        "assistant": "HumanMessage", 
    }),
    out = list(),
    next = ""
)

# ================================================================= NODES =================================================================

def conversation_node(state: State):
    ...

# ================================================================= GRAPH =================================================================

from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

BUILDER = StateGraph(State)
BUILDER.add_node(START, conversation_node)

MEMORY = MemorySaver()

GRAPH = BUILDER.compile(checkpointer=MEMORY)

# ================================================================= EXPORT ================================================================

__all__ = ['GRAPH', 'DEFAULT_STATE', 'State']
