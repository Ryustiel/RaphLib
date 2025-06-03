
from typing import (
    Any, Union, Dict, 
    List, Tuple, Literal, 
    Iterator, AsyncIterator, 
    Type, Optional, Callable,
)
from abc import ABC
import pydantic, asyncio, traceback

from langchain_core.tools import BaseTool as LangchainBaseTool
from langchain_core.messages import ToolMessage, ToolMessageChunk, ToolCall

from ..helpers import run_in_parallel_event_loop, get_or_create_event_loop


# TODO : BaseTool be called with a ToolCall object ;
# TODO : ToolKit would extract ToolCalls from an AIMessage and gather all responses from inner tools


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
    args_schema: Optional[Type[pydantic.BaseModel]] = None  # Arg schema if provided should be the only argument of _run methods

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        methods_to_check = ['_stream', '_astream', '_run', '_arun']
        available = []
        for name in methods_to_check:
            base_method = getattr(BaseTool, name)
            subclass_method = getattr(cls, name, None)
            # If the method was overridden (i.e. is a different function)
            if subclass_method is not None and subclass_method is not base_method:
                available.append(name)
        if not available:
            raise NotImplementedError(
                f"Subclass {cls.__name__} must override at least one of {methods_to_check}"
            )
        cls.available_methods = available

    def _extract_tool_call_id(self, mixed_parameters: Optional[Union[str, dict, ToolCall]]) -> str:
        if isinstance(mixed_parameters, dict) and "id" in mixed_parameters.keys():
            return mixed_parameters["id"]
        else:
            return "None"

    def _extract_parameters(self, mixed_parameters: Optional[Union[str, dict, ToolCall]]) -> Optional[pydantic.BaseModel]:
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

    # ================================================================= DEFAULT BEHAVIOR METHODS

    async def _arun(self, inp: Optional[pydantic.BaseModel] = None) -> Any:
        """
        Execute the tool and return the output or errors
        """
        if "_run" in self.available_methods:
            return self._run(inp=inp) if inp else self._run(inp)
        
        else:  # _astream or _stream
            event = None
            async_gen_instance = self._astream(inp=inp) if inp else self._astream()
            async for event in async_gen_instance:
                pass
            return event

    def _run(self, inp: Optional[pydantic.BaseModel] = None) -> Any:

        if "_arun" in self.available_methods:
            coro = self._arun(inp=inp) if inp else self._arun()
            if get_or_create_event_loop().is_running():
                return run_in_parallel_event_loop(future=coro)
            else:
                return asyncio.run(main=coro)

        else:  # _stream or _astream
            event = None
            gen_instance = self._stream(inp=inp) if inp else self._stream()
            for event in gen_instance:
                pass
            return event

    async def _astream(self, inp: Optional[pydantic.BaseModel] = None) -> AsyncIterator[Any]:
        """
        Streams the tool output as stream events, asynchronously (errors, partial responses, full response).
        """
        if "_stream" in self.available_methods:

            gen_instance = self._stream(inp=inp) if inp else self._stream()
            for event in gen_instance:
                yield event

        else:  # _arun or _run
            coro = self._arun(inp=inp) if inp else self._arun()
            yield await coro

    def _stream(self, inp: Optional[pydantic.BaseModel] = None) -> Iterator[Any]:
        """
        Streams the tool output as stream events (errors, partial responses, full response).
        """
        if "_astream" in self.available_methods:

            async_gen_instance = self._astream(inp=inp) if inp else self._astream()
            try:
                loop = get_or_create_event_loop()
                if loop.is_running():
                    while True: 
                        yield run_in_parallel_event_loop(future=async_gen_instance.__anext__())
                else:
                    while True: 
                        yield loop.run_until_complete(future=async_gen_instance.__anext__())
        
            except StopAsyncIteration:
                pass

        else:  # _run or _arun
            result = self._run(inp=inp) if inp else self._run()
            yield result

    # ================================================================= GATEWAY METHODS

    async def arun(self, mixed_parameters: Optional[Union[str, Dict[str, Any], ToolCall]] = None)  -> ToolMessage:
        """
        Execute the tool and return the output or errors
        """
        
        try:
            if self.args_schema:
                inp = self._extract_parameters(mixed_parameters=mixed_parameters)
                result = await self._arun(inp=inp)
            else:
                result = await self._arun()
            
            return ToolMessage(
                tool_call_id = self._extract_tool_call_id(mixed_parameters=mixed_parameters),
                status = "success",
                content = str(result),
            )
        
        except Exception as e:
            return ToolMessage(
                tool_call_id = self._extract_tool_call_id(mixed_parameters=mixed_parameters),
                status = "error",
                content = " ".join(traceback.format_exception(type(e), e, e.__traceback__))
            )
            
    def run(self, mixed_parameters: Optional[Union[str, Dict[str, Any], ToolCall]] = None) -> ToolMessage:
        """
        Execute the tool and return the output or errors.
        """
        try:
            if self.args_schema:
                inp = self._extract_parameters(mixed_parameters=mixed_parameters)
                result = self._run(inp=inp)
            else:
                result = self._run()
            
            return ToolMessage(
                tool_call_id = self._extract_tool_call_id(mixed_parameters=mixed_parameters),
                status = "success",
                content = str(result),
            )
        
        except Exception as e:
            return ToolMessage(
                tool_call_id = self._extract_tool_call_id(mixed_parameters=mixed_parameters),
                status = "error",
                content = " ".join(traceback.format_exception(type(e), e, e.__traceback__))
            )
            
    async def astream(self, mixed_parameters: Optional[Union[str, Dict[str, Any], ToolCall]] = None) -> AsyncIterator[ToolMessageChunk | ToolMessage]:
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
                
            tool_call_id = self._extract_tool_call_id(mixed_parameters=mixed_parameters)

            async for event in stream:
                
                yield ToolMessageChunk(
                    tool_call_id = tool_call_id,
                    status = "success",
                    content = str(event),
                )
        
        except Exception as e:
            yield ToolMessage(
                tool_call_id = self._extract_tool_call_id(mixed_parameters=mixed_parameters),
                status = "error",
                content = " ".join(traceback.format_exception(type(e), e, e.__traceback__))
            )
            
    def stream(self, mixed_parameters: Optional[Union[str, Dict[str, Any], ToolCall]] = None) -> Iterator[ToolMessageChunk | ToolMessage]:
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
                
            tool_call_id = self._extract_tool_call_id(mixed_parameters=mixed_parameters)

            for event in stream:
                
                yield ToolMessageChunk(
                    tool_call_id = tool_call_id,
                    status = "success",
                    content = str(event),
                )
        
        except Exception as e:
            yield ToolMessage(
                tool_call_id = self._extract_tool_call_id(mixed_parameters=mixed_parameters),
                status = "error",
                content = " ".join(traceback.format_exception(type(e), e, e.__traceback__))
            )
            
    def __str__(self):
        if self.args_schema is None:
            return f"\n=== Tool\nName: \n{self.name}\n\nModel: \nNo Input\n\nDescription: \n{self.description}\n=== End Tool"
        else:
            return f"\n=== Tool\nName: \n{self.name}\n\nModel: \n{self.args_schema.__name__} > {self.args_schema.model_fields}\n\nDescription: \n{self.description}\n=== End Tool"
