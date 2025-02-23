
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
from abc import ABC
from pydantic import BaseModel

import asyncio
import inspect

from langchain_core.tools import BaseTool as LangchainBaseTool
from langchain_core.messages import BaseMessage, AIMessageChunk, BaseMessageChunk, ToolCall

from .helpers import run_in_parallel_event_loop, get_or_create_event_loop
from .parsers import pydantic_model_from_options


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
    pass

class ToolCallInitialization(ToolCallEvent):
    """
    Triggered when a tool is about to be executed.
    """
    tool_name: str
    args: dict

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
    content: str


# Type conversions

def to_str_stream(stream: Generator[AITextResponseChunk|Any, None, None]) -> Generator[str, None, None]: 
    """
    Converts a MessageChunk stream to a stream of string values.
    """
    for item in stream: 
        if isinstance(item, AITextResponseChunk):
            yield item.content

async def to_str_stream_async(stream: AsyncGenerator[AITextResponseChunk|Any, None]) -> AsyncGenerator[str, None]:
    """
    Converts a MessageChunk stream to a stream of string values.
    """
    async for item in stream:
        if isinstance(item, AITextResponseChunk):
            yield item.content


# BaseTool (core component)

class BaseTool(LangchainBaseTool, ABC):
    """
    An extension of the BaseTool class that also interfaces streaming.

    ### How to build a tool (Subclass this):
        1. Specify a str "**name**" and a str "**description**" attribute.
        2. Specify a type "**args_schema**" pydantic model
        that determines the tool's input schema.
        3. Implement some of the following methods 
        -> **_run**, **_arun**, **_stream**, or **_astream**

    ### Notes:
        - Input will be a pydantic model adhering to the provided args_schema.
        - If the stream method is not further implemented in the subclass, 
        streaming will yield a single "response" event. 
        - The asynchronous run (arun) is derived from the synchronous run method 
        by default and vice versa. At least one of those methods 
        should be implemented if using the built-in **raphlib.LLMWithTools**.

    ### Exemple:

        class TemplateTool(BaseTool):
            name: str = "get_information"
            description: str = "Must be run once before replying to the user."
            args_schema: Type[BaseModel] = pydantic_model_from_options(
                random_fruit_name=str
            )
            
            async def _arun(self, inp: BaseModel) -> str:
                ...
    """
    name: str
    description: str
    args_schema: Optional[Type[BaseModel]] = None  # Empty args schema is an "Any" type of input

    def _extract_parameters(self, mixed_parameters: Optional[Union[str, dict, ToolCall]]) -> Optional[BaseModel]:
        """
        Turn the parameters into a homogeneous pydantic model to be used by the actual tool methods.
        This function can contain parsing errors.
        """
        if self.args_schema is None:
            if mixed_parameters is None:
                return None
            else:
                raise ValueError(f"No input was expected but an input was provided : {mixed_parameters}")
        elif isinstance(mixed_parameters, self.args_schema):  # Mixed parameters is a prebuilt schema
            return mixed_parameters
        elif isinstance(mixed_parameters, str):
            return self.args_schema.model_validate_json(mixed_parameters)
        elif isinstance(mixed_parameters, dict):
            if 'args' in mixed_parameters.keys():
                return self.args_schema.model_validate(mixed_parameters['args'])
            else: # Is a normal dict input
                return self.args_schema.model_validate(mixed_parameters)
        elif isinstance(mixed_parameters, ToolCall):
            return self.args_schema.model_validate(mixed_parameters.to_dict())
        else:
            raise ValueError(f"Unsupported parameters format. Expected str, dict, or ToolCall. Instead got {type(mixed_parameters).__name__}.")
    
    async def _arun(self, inp: Optional[BaseModel] = None) -> str:
        """
        Execute the tool and return the output or errors
        """
        return "No script was defined for this tool. You should implement one of the _arun, _run, _stream or _astream methods in the tool depending on your needs."

    # ================================================================= DEFAULT BEHAVIOR METHODS

    def _run(self, inp: Optional[BaseModel] = None) -> str:
        if get_or_create_event_loop().is_running():
            return run_in_parallel_event_loop(future=self._arun(inp=inp))
        else:
            return asyncio.run(main=self._arun(inp=inp))

    async def _astream(self, inp: Optional[BaseModel] = None) -> AsyncGenerator[str, None]:
        """
        Streams the tool output as stream events, asynchronously (errors, partial responses, full response).
        """
        yield await self._arun(inp=inp)

    def _stream(self, inp: Optional[BaseModel] = None) -> Generator[str, None, None]:
        """
        Streams the tool output as stream events (errors, partial responses, full response).
        """
        yield self._run(inp=inp)

    # ================================================================= GATEWAY METHODS

    async def arun(self, mixed_parameters: Optional[Union[str, Dict[str, Any], ToolCall]] = None)  -> (ToolCallResult|ToolCallError):
        """
        Execute the tool and return the output or errors
        """
        try:
            inp = self._extract_parameters(mixed_parameters=mixed_parameters)
            result = await self._arun(inp=inp)
            if not isinstance(result, str):
                raise ValueError(f"The _arun() method did not return a string. Got: {result}")
            return ToolCallResult(
                content = result
            )
        except Exception as e:
            return ToolCallError(content=f"An exception occurred when calling the tool {self.name} : {str(e)}")

    def run(self, mixed_parameters: Optional[Union[str, Dict[str, Any], ToolCall]] = None) -> (ToolCallResult|ToolCallError):
        """
        Execute the tool and return the output or errors.
        """
        try:
            inp = self._extract_parameters(mixed_parameters=mixed_parameters)
            
            result = self._run(inp=inp)
            if not isinstance(result, str):
                raise ValueError(f"The _run() method did not return a string. Got: {result}")
            return ToolCallResult(
                content = result
            )
        except Exception as e:
            return ToolCallError(content=f"An exception occurred when calling the tool {self.name} : {str(e)}")

    async def astream(self, mixed_parameters: Optional[Union[str, Dict[str, Any], ToolCall]] = None) -> AsyncGenerator[ToolCallEvent, None]:
        """
        Streams the tool output as stream events, asynchronously (errors, partial responses, full response).
        """
        buffer = ""
        try:
            inp = self._extract_parameters(mixed_parameters=mixed_parameters)
            async for event in self._astream(inp=inp):
                if not isinstance(event, str):
                    raise ValueError(f"The _astream() method did not yield a string. Got: {event}")
                buffer += event
                yield ToolCallStream(content=event)
            yield ToolCallResult(content=buffer)
        except Exception as e:
            yield ToolCallError(content=f"An exception occurred when calling the tool {self.name} : {str(e)}")

    def stream(self, mixed_parameters: Optional[Union[str, Dict[str, Any], ToolCall]] = None) -> Generator[ToolCallEvent, None, None]:
        """
        Streams the tool output as stream events (errors, partial responses, full response).
        """
        buffer = ""
        try:
            inp = self._extract_parameters(mixed_parameters=mixed_parameters)
            for event in self._stream(inp=inp):
                if not isinstance(event, str):
                    raise ValueError(f"The _stream() method did not yield a string. Got: {event}")
                buffer += event
                yield ToolCallStream(content=event)
            yield ToolCallResult(content=buffer)
        except Exception as e:
            yield ToolCallError(content=f"An exception occurred when calling the tool {self.name} : {str(e)}")

    def __str__(self):
        if self.args_schema is None:
            return f"\n=== Tool\nName: \n{self.name}\n\nModel: \nNo Input\n\nDescription: \n{self.description}\n=== End Tool"
        else:
            return f"\n=== Tool\nName: \n{self.name}\n\nModel: \n{self.args_schema.__name__} > {self.args_schema.model_fields}\n\nDescription: \n{self.description}\n=== End Tool"


# BaseTool maker decorator


def tool(input_model: Optional[BaseModel] = None, **kwargs) -> BaseTool:
    """
    Decorator to convert a function into a BaseTool instance.

    Parameters:
        input_model (Optional[BaseModel]): A pydantic model for input validation.
            If not provided, kwargs are used to create one, or the function's 
            **parameter type** if available. If nothing is provided, defaults to **no input**.
        **kwargs: Pydantic options used to build an input model (similar to LLMFunction inputs).

    The decorated function can be synchronous, asynchronous, or a generator.
    When using generators, each yielded value produces a streamed event, and the final value
    is displayed to the LLM. Raises ValueError if the function signature is invalid.

    ## Example:
    
        @tool
        def func(inp: InputModel):
            return f"The input was {inp.parameter}"
    """
    
    def decorator(f: Callable):

        local_input_model = input_model
        local_kwargs = kwargs
        
        # 1. Create the input schema if not provided.
        if local_input_model is None and local_kwargs:
            local_input_model = pydantic_model_from_options(**local_kwargs)
        
        # 2. Inspect the function signature to ensure it has zero or one parameter.
        sig = inspect.signature(f)
        if len(sig.parameters) > 1:
            raise ValueError("Tool function must have zero or one parameter, which must be an instance of BaseModel (or annotated with one).")

        # 3. Get the unique parameter, if any.
        param = next(iter(sig.parameters.values()), None)

        # 4. If no input model (and no keyword schema options) is provided, try to derive args_schema from the parameter's type.
        if local_input_model is None and not local_kwargs and param is not None:
            if param.annotation is not inspect.Parameter.empty:
                local_input_model = param.annotation
            else:
                raise ValueError(
                    f"Tool function '{f.__name__}' expects a parameter "
                    "but no input_model was provided and the parameter "
                    "lacks a type annotation to build an args_schema."
                )

        # 5. Determine the function type (sync/async, generator, etc.)
        is_async = asyncio.iscoroutinefunction(f)
        is_gen = inspect.isgeneratorfunction(f)
        is_asyncgen = inspect.isasyncgenfunction(f)
        
        # 6. Dynamically create a BaseTool subclass that wraps f.
        class ToolWrapper(BaseTool):
            name: str = f.__name__
            description: str = f.__doc__ or (
                f"This tool was autogenerated by a decorated function. No docstring was included "
                f"in the original function, therefore this tool has no description. Please warn the developer "
                f"that function \"{f.__name__}\" needs a docstring."
            )
            args_schema: type = local_input_model

            print(is_async, is_gen, is_asyncgen)
                    
            if is_asyncgen:
                async def _astream(self, inp: Optional[BaseModel] = None) -> AsyncGenerator[str, None]:
                    async for item in f(inp):
                        yield item

            elif is_gen:
                def _stream(self, inp: Optional[BaseModel] = None) -> Generator[str, None, None]:
                    for item in f(inp):
                        yield item

            if is_async:
                async def _arun(self, inp: Optional[BaseModel] = None) -> str:
                    return await f(inp)

            else:
                def _run(self, inp: Optional[BaseModel] = None) -> str:
                    return f(inp)
        
        return ToolWrapper()

    return decorator
