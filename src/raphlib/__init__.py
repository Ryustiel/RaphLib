"""
A library to handle a LLM Conversation.
"""

__author__ = "Raphael Nguyen"
__copyright__ = "© 2025 Raphael Nguyen"
__license__ = "MIT"
__version__ = "1.0.0"

import langchain_core, pydantic_core, langchain, pydantic  # Dependencies TODO : Manage package
from langchain_core.messages import AIMessageChunk, AIMessage
# Standard Library : os, typing, logging, asyncio

from .chat import ChatHistory, ChatMessage, LangchainMessageTypes

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

from .tools_old import (
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

from . import graph, tools

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

    'graph',
]
