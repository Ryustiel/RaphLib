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
    chat: ChatHistory
    next: str
    out: List[str]

DEFAULT_STATE = State(
    chat = ChatHistory(types={
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
    response = (
        state.chat.using_prompt("system", 
            """
            You are bored. You may chat with the members of the conversation but you really don't care. 
            """)
        | LLM
    ).invoke({})
    return {
        "chat": state.chat.append("ai", response.content),
        "out": [response.content]
    }

# ================================================================= GRAPH =================================================================

from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

BUILDER = StateGraph(State)
BUILDER.add_node("Chat", conversation_node)

BUILDER.add_edge(START, "Chat")
BUILDER.add_edge("Chat", "Chat")

MEMORY = MemorySaver()

GRAPH = BUILDER.compile(checkpointer=MEMORY)

# ================================================================= EXPORT ================================================================

__all__ = ['GRAPH', 'DEFAULT_STATE', 'State']
