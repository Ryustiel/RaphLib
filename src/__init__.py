"""
A library to handle a LLM Conversation.
"""

import dotenv, langchain_core, pydantic_core, langchain, pydantic, langchain_openai  # Dependencies TODO : Manage package
# Standard Library : os, typing, logging, asyncio

from .prompts import ChatHistory, ChatMessage
from .functions import LLMFunction, LLMFunctionResult, BatchLLMFunctionResult
from .tools import LLMWithTools, ToolInterrupt, StreamableBaseTool
from .stables import StableModel, pydantic_model_from_options
from .helpers import (
    LapTimer, 
    balance_results, 
    first_completed, 
    escape_characters, 
    get_all_fields_as_optional, 
    repair_json,
    repair_list,
    diff_dict,
)

__all__ = [
    'ChatHistory', 
    'ChatMessage',

    'LLMFunction', 
    'LLMFunctionResult', 
    'BatchLLMFunctionResult', 
    'LLMWithTools', 

    'StableModel',
    
    'ToolInterrupt', 
    'StreamableBaseTool',
    'pydantic_model_from_options',

    'balance_results',
    'first_completed',
    'escape_characters',
    'get_all_fields_as_optional',
    'repair_json',
    'repair_list',
    'diff_dict',
    'LapTimer',
]
