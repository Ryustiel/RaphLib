"""
A library to handle a LLM Conversation.
"""

import dotenv, langchain_core, pydantic_core, langchain, pydantic, langchain_openai  # Dependencies TODO : Manage package
# Standard Library : os, typing, logging, asyncio

from .prompts import ChatHistory, ChatMessage
from .functions import LLMFunction, LLMFunctionResult, BatchLLMFunctionResult
from .tools import LLMWithTools, ToolInterrupt
from .helpers import balance_results, first_completed, escape_characters, LapTimer
from .setup import setup_env

__all__ = [
    'ChatHistory', 
    'ChatMessage',
    'LLMFunction', 
    'LLMFunctionResult', 
    'BatchLLMFunctionResult', 
    'LLMWithTools', 
    'ToolInterrupt', 
    'balance_results',
    'first_completed',
    'setup_env',
    'escape_characters',
    'LapTimer',
]
