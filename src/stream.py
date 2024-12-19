from typing import (
    Any, 
    Union, 
    Dict, 
    List, 
    Tuple, 
    Literal, 
    Generator, 
    AsyncGenerator,
    Type, 
    Optional, 
    get_args,
    get_origin,
)
from pydantic import BaseModel, Field, create_model
from pydantic_core import ValidationError


StreamFlags = Literal["cancel", "interrupt"]


class StreamEvent(BaseModel):
    """
    An event that can be outputted by a stream.
    """
    pass


class ResetStream(StreamEvent):
    """
    Flag raised when a stream has been reset.
    """
    error: str

class ToolCallInitialization(StreamEvent):
    """
    Triggered when a tool is about to be executed.
    """
    pass

class ToolCallResult(StreamEvent):
    """
    Triggered when a tool is done with a result.
    """
    pass

class ToolCallError(StreamEvent):
    """
    Triggered when a tool has been called and returned an error.
    """
    pass
