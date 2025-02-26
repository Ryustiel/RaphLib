
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
from .events import *


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
            args_schema: type = pydantic_model_from_options(
                random_fruit_name=str
            )
            
            async def _arun(self, inp: BaseModel) -> str:
                ...
    """
    name: str
    description: str
    args_schema: Optional[Type[BaseModel]] = None  # Arg schema if provided should be the only argument of _run methods

    def _extract_parameters(self, mixed_parameters: Optional[Union[str, dict, ToolCall]]) -> Optional[BaseModel]:
        """
        Turn the parameters into a homogeneous pydantic model to be used by the actual tool methods.
        This function can contain parsing errors.
        """
        if isinstance(mixed_parameters, self.args_schema):  # Mixed parameters is a prebuilt schema
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
        raise NotImplementedError("""
            This tool lacks the appropriate method for the invoke context. 
            You should implement one of the _arun, _run, _stream or _astream methods in the tool depending on your needs.
            This error might also occur if you only defined _stream() or _astream() without defining fallback _run or _arun 
            while calling an async ainvoke or sync invoke. _stream and _astream do not convert into one another automatically,
            unlike _run and _arun.
        """)

    # ================================================================= DEFAULT BEHAVIOR METHODS

    def _run(self, inp: Optional[BaseModel] = None) -> str:
        coro = self._arun(inp=inp) if inp else self._arun()
        if get_or_create_event_loop().is_running():
            return run_in_parallel_event_loop(future=coro)
        else:
            return asyncio.run(main=coro)

    async def _astream(self, inp: Optional[BaseModel] = None) -> AsyncGenerator[str, None]:
        """
        Streams the tool output as stream events, asynchronously (errors, partial responses, full response).
        """
        coro = self._arun(inp=inp) if inp else self._arun()
        yield await coro

    def _stream(self, inp: Optional[BaseModel] = None) -> Generator[str, None, None]:
        """
        Streams the tool output as stream events (errors, partial responses, full response).
        """
        result = self._run(inp=inp) if inp else self._run()
        yield result

    # ================================================================= GATEWAY METHODS

    async def arun(self, mixed_parameters: Optional[Union[str, Dict[str, Any], ToolCall]] = None)  -> (ToolCallResult|ToolCallError):
        """
        Execute the tool and return the output or errors
        """
        try:
            if self.args_schema:
                inp = self._extract_parameters(mixed_parameters=mixed_parameters)
                result = await self._arun(inp=inp)
            else:
                result = await self._arun()

            if not isinstance(result, str):
                raise ValueError(f"The _arun() method did not return a string. Got: {result}")
            return ToolCallResult(content=result, tool_name=self.name)
        
        except BaseInterrupt as interrupt:
            raise interrupt
        
        except Exception as e:
            return ToolCallError(content=f"An exception occurred when calling the tool {self.name} : {type(e).__name__} {str(e)}", tool_name=self.name)

    def run(self, mixed_parameters: Optional[Union[str, Dict[str, Any], ToolCall]] = None) -> (ToolCallResult|ToolCallError):
        """
        Execute the tool and return the output or errors.
        """
        try:
            if self.args_schema:
                inp = self._extract_parameters(mixed_parameters=mixed_parameters)
                result = self._run(inp=inp)
            else:
                result = self._run()

            if not isinstance(result, str):
                raise ValueError(f"The _run() method did not return a string. Got: {result}")
            return ToolCallResult(content=result, tool_name=self.name)
        
        except BaseInterrupt as interrupt:
            raise interrupt
        
        except Exception as e:
            return ToolCallError(content=f"An exception occurred when calling the tool {self.name} : {type(e).__name__} {str(e)}", tool_name=self.name)

    async def astream(self, mixed_parameters: Optional[Union[str, Dict[str, Any], ToolCall]] = None) -> AsyncGenerator[ToolCallEvent, None]:
        """
        Streams the tool output as stream events, asynchronously (errors, partial responses, full response).
        """
        buffer = "Generator Tool Trace\n<start>\n"
        try:
            if self.args_schema:
                inp = self._extract_parameters(mixed_parameters=mixed_parameters)
                stream = self._astream(inp=inp)
            else:
                stream = self._astream()

            async for event in stream:
                if not isinstance(event, str):
                    raise ValueError(f"The _astream() method did not yield a string. Got: {event}")
                buffer += event + "\n<next>\n"
                yield ToolCallStream(content=event, tool_name=self.name)
            yield ToolCallResult(content=buffer, tool_name=self.name)

        except BaseInterrupt as interrupt:
            raise interrupt
        
        except Exception as e:
            yield ToolCallError(content=f"An exception occurred when calling the tool {self.name} : {type(e).__name__} {str(e)}", tool_name=self.name)

    def stream(self, mixed_parameters: Optional[Union[str, Dict[str, Any], ToolCall]] = None) -> Generator[ToolCallEvent, None, None]:
        """
        Streams the tool output as stream events (errors, partial responses, full response).
        """
        buffer = "Generator Tool Trace\n<start>\n"
        try:
            if self.args_schema:
                inp = self._extract_parameters(mixed_parameters=mixed_parameters)
                stream = self._stream(inp=inp)
            else:
                stream = self._stream()

            for event in stream:
                if not isinstance(event, str):
                    raise ValueError(f"The _stream() method did not yield a string. Got: {event}")
                buffer += event + "\n<next>\n"
                yield ToolCallStream(content=event, tool_name=self.name)
            yield ToolCallResult(content=buffer, tool_name=self.name)

        except BaseInterrupt as interrupt:
            raise interrupt
        
        except Exception as e:
            yield ToolCallError(content=f"An exception occurred when calling the tool {self.name} : {type(e).__name__} {str(e)}", tool_name=self.name)

    def __str__(self):
        if self.args_schema is None:
            return f"\n=== Tool\nName: \n{self.name}\n\nModel: \nNo Input\n\nDescription: \n{self.description}\n=== End Tool"
        else:
            return f"\n=== Tool\nName: \n{self.name}\n\nModel: \n{self.args_schema.__name__} > {self.args_schema.model_fields}\n\nDescription: \n{self.description}\n=== End Tool"


# BaseTool maker decorator


def tool(input_model: Callable | Optional[BaseModel] = None, **kwargs) -> BaseTool:
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
    # @tool has two modes : factory mode and decorator mode (flagged by "input_function")
    if callable(input_model):

        input_function = input_model
        input_model = None
        kwargs = {}
        
    else:
        input_function = None
    
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
                    f"Tool function '{f.__name__}' expects a Type[BaseModel] parameter. "
                    "You can provide the input_model as a parameter of the @tool decorator "
                    "or as a type annotation on that first parameter."
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
                    
            if is_asyncgen:
                async def _astream(self, inp: Optional[BaseModel] = None) -> AsyncGenerator[str, None]:
                    stream = f(inp) if inp else f()
                    async for item in stream:
                        yield item

            elif is_gen:
                def _stream(self, inp: Optional[BaseModel] = None) -> Generator[str, None, None]:
                    stream = f(inp) if inp else f()
                    for item in stream:
                        yield item

            if is_async:
                async def _arun(self, inp: Optional[BaseModel] = None) -> str:
                    coro = f(inp) if inp else f()
                    return await coro

            else:
                def _run(self, inp: Optional[BaseModel] = None) -> str:
                    result = f(inp) if inp else f()
                    return result
        
        return ToolWrapper()
    

    if input_function:  # Decorator mode
        return decorator(input_function)
        
    else:  # Factory mode
        return decorator
