"""
A library to handle a LLM Conversation.
"""

import dotenv, langchain_core, pydantic_core, langchain, pydantic, langchain_openai  # Dependencies TODO : Manage package
from langchain_core.messages import AIMessageChunk, AIMessage
# Standard Library : os, typing, logging, asyncio

from .prompts import ChatHistory, ChatMessage, LangchainMessageTypes

from .functions import LLMFunction, LLMFunctionResult, BatchLLMFunctionResult

from .stables import StableModel, pydantic_model_from_options

from .events import (
    StreamEvent, 
    ResetStream, 
    ToolCallEvent, 
    ToolCallError, 
    ToolCallInitialization, 
    ToolCallStream, 
)

from .stream import LLMWithTools, ToolInterrupt, BaseInterrupt

from .tools import (
    to_str_stream,
    to_str_stream_async,
    tool,
    BaseTool,
)

from .helpers import (
    LapTimer, 
    balance_results, 
    first_completed, 
    escape_characters, 
    get_all_fields_as_optional, 
    repair_json,
    repair_list,
    diff_dict,
    run_in_parallel_event_loop,
    get_or_create_event_loop,
)

__all__ = [
    'AIMessageChunk',
    'AIMessage',

    'ChatHistory', 
    'ChatMessage',
    'LangchainMessageTypes',

    'LLMFunction', 
    'LLMFunctionResult', 
    'BatchLLMFunctionResult', 
    'LLMWithTools', 

    'StableModel',
    
    'ToolInterrupt', 
    'BaseInterrupt',
    'pydantic_model_from_options',

    'balance_results',
    'first_completed',
    'escape_characters',
    'get_all_fields_as_optional',
    'run_in_parallel_event_loop',
    'get_or_create_event_loop',
    'repair_json',
    'repair_list',
    'diff_dict',
    'LapTimer',

    'StreamEvent', 
    'ResetStream', 
    'ToolCallEvent', 
    'ToolCallError', 
    'ToolCallInitialization', 
    'ToolCallStream',

    'BaseTool',
    'tool',

    'to_str_stream',
    'to_str_stream_async',
]
