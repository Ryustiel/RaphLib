
from typing import (
    Literal,
    List,
)
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk

import traceback


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

class AITextResponseChunk(StreamEvent):
    """
    Contains bits of the text response of the model.
    Should be deleted if the ResetStream event is received.
    """
    content: str

class AITextResponse(AIMessageChunk):
    """
    Represents completed AIMessageChunks that a LLMWithTools has emitted.
    Behaves like an AIMessage.
    """
    pass

class ToolCallEvent(StreamEvent):
    """
    Base class for all events that were triggered by a tool call.
    """
    content: str = ""
    pass

class ToolCallInitialization(ToolCallEvent):
    """
    Triggered when a tool is about to be executed.
    """
    tool_name: str
    args: dict

    @property
    def content(self):
        return f"Calling Tool {self.tool_name} with {self.args}"

class ToolCallStream(ToolCallEvent):
    """
    Triggered when a result is being streamed from the tool.
    """
    tool_name: str
    content: str

class ToolCallResult(ToolCallEvent):
    """
    Triggered when a tool is done with a result.
    """
    tool_name: str
    content: str

class ToolCallError(ToolCallEvent):
    """
    Triggered when a tool has been called and returned an error.
    """
    tool_name: str
    content: str

class BaseInterrupt(BaseException):
    """
    A class of exceptions to pause the normal flow of the llm exchange process.
    """
    pass

class ToolInterrupt(BaseInterrupt):
    def __init__(self, tool_result: str, *args):
        """
        Describe information about a tool that required the tool calling process to pause.
        This object holds information to resume the tool calling process.
        """
        self.tool_result: str = tool_result
        
        # Information for resuming tool calling
        self.__tool_name: str = None
        self.__messages: List[BaseMessage] = None
        self.__llm_response: AIMessage = None
        self.__current_tool_index: int = None
        self.__n_tool_called: int = None

        super().__init__(f"ToolInterrupt: {tool_result}", *args)

    def add_stream_metadata(self, tool_name: str, messages: List[BaseMessage], llm_response: AIMessage, current_tool_index: int, n_tool_called: int):
        self.__tool_name = tool_name
        self.__messages = messages
        self.__llm_response = llm_response
        self.__current_tool_index = current_tool_index
        self.__n_tool_called = n_tool_called
    
    @property
    def tool_name(self) -> str:
        if self.__tool_name is None:
            tb = "".join(traceback.format_stack())
            raise ValueError(f"__tool_name is None (value: {self.__tool_name}). Traceback:\n{tb}")
        return self.__tool_name

    @property
    def messages(self) -> List[BaseMessage]:
        if self.__messages is None:
            tb = "".join(traceback.format_stack())
            raise ValueError(f"__messages is None (value: {self.__messages}). Traceback:\n{tb}")
        return self.__messages

    @property
    def llm_response(self) -> AIMessage:
        if self.__llm_response is None:
            tb = "".join(traceback.format_stack())
            raise ValueError(f"__llm_response is None (value: {self.__llm_response}). Traceback:\n{tb}")
        return self.__llm_response
    
    @property
    def current_tool_index(self) -> int:
        if self.__current_tool_index is None:
            tb = "".join(traceback.format_stack())
            raise ValueError(f"__current_tool_index is None (value: {self.__current_tool_index}). Traceback:\n{tb}")
        return self.__current_tool_index

    @property
    def n_tool_called(self) -> int:
        if self.__n_tool_called is None:
            tb = "".join(traceback.format_stack())
            raise ValueError(f"__n_tool_called is None (value: {self.__n_tool_called}). Traceback:\n{tb}")
        return self.__n_tool_called

    def __repr__(self):
        """Optional: custom string representation of the ToolInterrupt."""
        return f"<ToolInterrupt tool_name={self.__tool_name}, tool_result={self.tool_result}>"
    