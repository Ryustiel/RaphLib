import logging

from typing import List, Tuple
from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI

from src import LLMFunction, ChatHistory, LLMWithTools, LLMFunctionResult, setup_env

# ================================================================= TOOLS =================================================================

@tool
def mystery_operation(a: int, b: int) -> Tuple[int, str]:
    """Performs the mystery operation on the two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return 2 * a + b, "blah"

@tool
def talked_about_fruit_event() -> str:
    """Returns a message about the fruit event."""
    return "User talked about the fruit."

# ================================================================= LLMS ==================================================================

setup_env()

LLM_MYSTERY_OPERATION = LLMWithTools(
    AzureChatOpenAI(
        deployment_name="gpt-4o",
        temperature=0.2,
        max_tokens=500,
    ),
    tools = [mystery_operation, talked_about_fruit_event],  
    tool_events = ["talked_about_fruit_event"],
    track_call = True,
)

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
        "summary": "SystemMessage", 
        "assistant": "HumanMessage", 
        "user": "HumanMessage"}),
    out = list(),
    next = ""
)

# ================================================================= NODES =================================================================

def conversation_node(state: State):
    state.CHAT.appendleft("temp", "You are a bored agent, tell the user to get fucked but don't use curse words. Just be very mean.")
    messages = (state.CHAT | LLM_MYSTERY_OPERATION).invoke({})
    state.CHAT.append(messages)
    state.CHAT.remove(["temp"])
    return {"CHAT": state.CHAT, "out": [state.CHAT.messages[-1].content]}

def check_node(state: State):
    response: LLMFunctionResult = (
        state.CHAT.using({"ai": "assistant", "human": "user"})
        | LLMFunction(LLM, "Write True if the user mentioned toilet seats, else False.", handle_error=False,  mentioned_toilet_seats=bool)
    ).invoke({})
    if response.success:
        return {
            "CHAT": state.CHAT, 
            "next": "quit" if response.result.mentioned_toilet_seats else "chat"
        }
    else:
        state.CHAT
        return {
            "CHAT": state.CHAT.appendleft("temp", 
                """
                The user triggered openai's security checks using an offensive prompt. 
                Enjoy telling them that this failed and that they are a failure.
                """), 
            "next": "chat"
        }

def quit_node(state: State):
    message = (
        state.CHAT.using_prompt("system", 
        """
        Now you suddenly have an irrepressible urge to talk about chocolate and you don't want to talk about anything else other than chocolate. 
        You still remember the beginning of the conversation though. If asked you can do the mystery operation.
        """)
        | LLM).invoke({}).content
    return {
        "CHAT": state.CHAT.remove(["temp"]).append("ai", message), 
        "out": [message]
    }

def summarizer_node(state: State):
    print("LEN CHAT", state.CHAT.without(["system", "summary"]).size(),  state.CHAT.size())

    if len(state.CHAT.without(["system", "summary"]).messages) > 10:
        response = (
            state.CHAT.using({"ai": "assistant", "human": "user"})
            | LLMFunction(LLM, "Summarize the conversation. When the conversation already contains summaries, the summary you output should be a concise continuation of the previous one.",  summary=str)
            ).invoke({})
        try:
            state.CHAT.overwrite(("summary", response.summary), 0, len(state.CHAT.messages) - 2, ignore_type=["summary", "system"])
        except Exception as e:
            print("ERROR\n", e)
        return {
            "CHAT": state.CHAT,
        }
    else:
        print("\n\nChat History:\n")
        for message in state.CHAT.messages:
            print(message.type, ">", message.content)
        return {"next": "chat"}

# ================================================================= GRAPH =================================================================

from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

BUILDER = StateGraph(State)
BUILDER.add_node("Chat", conversation_node)
BUILDER.add_node("Check", check_node)
BUILDER.add_node("Quit", quit_node)
BUILDER.add_node("Summarizer", summarizer_node)

BUILDER.add_edge(START, "Chat")
BUILDER.add_edge("Chat", "Check")
BUILDER.add_conditional_edges("Check", lambda state: state.next, {"quit": "Quit", "chat": "Summarizer"})
BUILDER.add_edge("Summarizer", "Chat")
BUILDER.add_edge("Quit", "Quit")

MEMORY = MemorySaver()

GRAPH = BUILDER.compile(checkpointer=MEMORY)

# ================================================================= EXPORT ================================================================

__all__ = ['GRAPH', 'DEFAULT_STATE', 'State']
