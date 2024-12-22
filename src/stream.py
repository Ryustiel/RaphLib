import asyncio

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
    Callable,
)
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, create_model
from pydantic_core import ValidationError
from langchain_core.tools import BaseTool
from langchain_core.messages import ToolCall


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


class ToolCallEvent(StreamEvent):
    """
    Base class for all events that were triggered by a tool call.
    """
    pass

class ToolCallInitialization(ToolCallEvent):
    """
    Triggered when a tool is about to be executed.
    """
    pass

class ToolCallStream(ToolCallEvent):
    """
    Triggered when a result is being streamed from the tool.
    """
    content: str

class ToolCallResult(ToolCallEvent):
    """
    Triggered when a tool is done with a result.
    """
    content: str

class ToolCallError(ToolCallEvent):
    """
    Triggered when a tool has been called and returned an error.
    """
    error_message: str


class StreamableBaseTool(BaseTool, ABC):
    """
    An extension of the BaseTool class that also interfaces streaming.
    If the stream method is not further implemented in the subclass, streaming will yield a single "response" event.
    arun is derived from the run method by default and vice versa. At least one of them should be implemented.
    
    General information on building tool : 
    1. Write a str "name" and a str "description" attributes.
    2. Add an args_schema pydantic model that will determine the input schema of the tool.
    3. Create at least an [async _arun(input)] method. You can also implement the _run, _stream, and _astream methods.
    
    NOTE : Input will contain a pydantic model with the input schema.
    NOTE : Unimplemented run and stream methods will link back to the _arun() method. 
    """
    def _extract_parameters(self, mixed_parameters: Union[str, dict, ToolCall]) -> BaseModel:
        """
        Turn the parameters into a homogeneous pydantic model to be used by the actual tool methods.
        This function can contain parsing errors.
        """
        print("RECEIVED PARAMETERS\n\n", type(mixed_parameters), "\n\n", mixed_parameters, "\n\n")
        assert False
        if isinstance(mixed_parameters, str):
            return self.args_schema.model_validate_json(mixed_parameters)
        elif isinstance(mixed_parameters, dict):
            return self.args_schema.model_validate(mixed_parameters)
        elif isinstance(mixed_parameters, ToolCall):
            return self.args_schema.model_validate(mixed_parameters.to_dict())
        else:
            raise ValueError(f"Unsupported parameters format. Expected str, dict, or ToolCall. Instead got {type(mixed_parameters).__name__}.")
    
    @abstractmethod
    async def _arun(self, inp: BaseModel) -> str:
        """
        Execute the tool and return the output or errors
        """
        pass

    # ================================================================= DEFAULT BEHAVIOR METHODS

    def _run(self, inp: BaseModel) -> str:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.run_coroutine_threadsafe(loop=loop, coro=self._arun(inp=inp))
        else:
            return asyncio.run(loop=loop, main=self._arun(inp=inp))

    async def _astream(self, inp: BaseModel) -> AsyncGenerator[str, None]:
        """
        Streams the tool output as stream events, asynchronously (errors, partial responses, full response).
        """
        yield await self._arun(inp=inp)

    def _stream(self, inp: BaseModel) -> Generator[str, None, None]:
        """
        Streams the tool output as stream events (errors, partial responses, full response).
        """
        yield self._run(inp=inp)

    # ================================================================= GATEWAY METHODS

    async def arun(self, mixed_parameters: Union[str, Dict[str, Any], ToolCall])  -> ToolCallEvent:
        """
        Execute the tool and return the output or errors
        """
        try:
            inp = self._extract_parameters(parameters=mixed_parameters)
            return ToolCallResult(
                content = await self._arun(inp=inp)
            )
        except Exception as e:
            return ToolCallError(error=f"An exception occurred when calling the tool {self.name} : {str(e)}")

    def run(self, mixed_parameters: Union[str, Dict[str, Any], ToolCall]) -> ToolCallEvent:
        """
        Execute the tool and return the output or errors.
        """
        try:
            inp = self._extract_parameters(parameters=mixed_parameters)
            return ToolCallResult(
                content = self._run(inp=inp)
            )
        except Exception as e:
            return ToolCallError(error=f"An exception occurred when calling the tool {self.name} : {str(e)}")

    async def astream(self, mixed_parameters: Union[str, Dict[str, Any], ToolCall]) -> AsyncGenerator[ToolCallEvent, None]:
        """
        Streams the tool output as stream events, asynchronously (errors, partial responses, full response).
        """
        buffer = ""
        try:
            inp = self._extract_parameters(parameters=mixed_parameters)
            async for event in self._astream(inp=inp):
                buffer += event
                yield ToolCallStream(content=event)
        except Exception as e:
            return ToolCallError(error=f"An exception occurred when calling the tool {self.name} : {str(e)}")
        
        yield ToolCallResult(content=buffer)

    def stream(self, mixed_parameters: Union[str, Dict[str, Any], ToolCall]) -> Generator[ToolCallEvent, None, None]:
        """
        Streams the tool output as stream events (errors, partial responses, full response).
        """
        buffer = ""
        try:
            inp = self._extract_parameters(parameters=mixed_parameters)
            for event in self._stream(inp=inp):
                buffer += event
                yield ToolCallStream(content=event)
        except Exception as e:
            return ToolCallError(error=f"An exception occurred when calling the tool {self.name} : {str(e)}")

        yield ToolCallResult(content=buffer)
