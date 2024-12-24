import logging
import asyncio

from typing import (
    Any, 
    List, 
    Dict,
    Union, 
    Optional, 
    Generator,
    AsyncGenerator,
)
from pydantic import BaseModel

from langchain_core.tools import StructuredTool
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel

from .helpers import escape_characters, run_in_parallel_event_loop
from .stream import BaseTool, StreamEvent, AITextResponseChunk, AITextResponse, StreamFlags, ToolCallError, ToolCallInitialization, ToolCallResult

class ToolInterrupt(BaseException):
    def __init__(self, tool_name: str, *args, **kwargs):
        self.tool_name: str = tool_name
        self.kwargs: dict = kwargs
        super().__init__(f"Event {self.tool_name} was triggered. Value is {kwargs}.", *args)

    def __getitem__(self, key):
        """Retrieve values from kwargs like a dict"""
        if key in self.kwargs:
            return self.kwargs[key]
        else:
            raise KeyError(f"Key '{key}' not found in ToolInterrupt kwargs.")
    
    def __repr__(self):
        """Optional: custom string representation of the ToolInterrupt."""
        return f"<ToolInterrupt tool_name={self.tool_name}, kwargs={self.kwargs}>"


class LLMWithTools(Runnable[LanguageModelInput, BaseMessage]):
    """
    A non serializable runnable that behaves like a BaseChatModel when invoked, but manages tools automatically.
    """
    def __init__(
            self, 
            llm: BaseChatModel, 
            tools: List[BaseTool], 
            interruptions: List[str] = [], 
            max_retries: int = 2,
            max_tool_depth: int = 10,

        ):
        """
        This class behaves like a BaseChatModel except for the fact that it 
        handles tool calls automatically, in different ways depending on the method you invoke it with.
        Can be sync and async streamed, or invoked.

        NOTE : If you are registering interruptions (len(self.interruptions) > 0), you have to handle ToolInterrupt exceptions.

        Tool Interrupts : 
            ToolInterrupts are specific events (the only events) that are triggered
            whenever a tool in self.interruption has been called by the LLM.
            The tool function is actually executed and its output is contained in the ToolInterrupt object.

            NOTE : This is designed to allow for extra processing of this tool call outside the LLMWithTool's invoke loop.
            This is espacially useful when the tool is affecting other parts of the code, its execution is not self contained.
            NOTE : You should be handling ToolInterrupt exceptions when you are registering interruptions.
        """
        self.tools = {tool.name: tool for tool in tools}
        self.interruptions = interruptions
        self.llm: BaseChatModel = llm.bind_tools(tools)
        self.max_retries = max_retries
        self.max_tool_depth = max_tool_depth

    def _create_error_tracking_message(self, tool_call: Dict[str, Any], error: Exception) -> str:
        """
        Create a system message with the tool call and result.
        """
        return f"""
            Tool call: {tool_call['name']} {escape_characters(str(tool_call['args']))}\n
            Error: {escape_characters(str(error))}
        """
    
    def _update_messages(self, messages: List[BaseMessage], tool_call_id: str, event: StreamEvent|AIMessage) -> List[BaseMessage]:
        """
        Updates the messages list depending on the event.
        """
        if isinstance(event, ToolCallResult):
            messages.append(ToolMessage(content=event.content, tool_call_id=tool_call_id))
        elif isinstance(event, ToolCallError):
            messages.append(SystemMessage(content=event.content))

    async def astream(self,
        input: LanguageModelInput, 
        config: Optional[RunnableConfig] = None, 
        sync_mode: bool = False,
        stream_text: bool = True,
        **kwargs
    ) -> AsyncGenerator[AITextResponse|StreamEvent, StreamFlags]:
        """
        Stream the llm output as well as special events when tools are called.

        sync_mode : whether to use the streaming's async functions or not.
        stream_text : whether to yield chunks of llm response obtained via stream or to yield entire str responses.
        NOTE : If stream_text is False then the streaming will yield full text messages (str) and StreamEvents.
        If it is set to True, the streaming will yield text message chunks (str) and StreamEvents.
        NOTE : stream_text can only be set to True if the LLM supports the .stream or .astream methods.
        """
        prompt: ChatPromptValue = self.llm._convert_input(input)
        messages: List[BaseMessage] = prompt.to_messages()  # Extract the messages from the prompt so it's compatible with llm invoke.

        n_tool_called: int = 0
        keep_streaming: bool = True
        while keep_streaming:
            n_consecutive_errors: int = 0

            if sync_mode:
                if stream_text:
                    # Streaming llm output unless it's a ToolCall
                    llm_response = AITextResponse(content='')
                    for chunk in self.llm.stream(messages, config, **kwargs):
                        llm_response += chunk
                        if chunk.content:
                            yield AITextResponseChunk(content=chunk.content)  # Yield the text content of this chunk
                else:
                    llm_response: AIMessage = self.llm.invoke(messages, config, **kwargs)
            else:
                if stream_text:
                    llm_response = AITextResponse(content='')
                    async for chunk in self.llm.astream(messages, config, **kwargs):
                        llm_response += chunk
                        if chunk.content:
                            yield AITextResponseChunk(content=chunk.content)  # Yield the text content of this chunk
                else:
                    llm_response: AIMessage = await self.llm.ainvoke(messages, config, **kwargs)

            messages.append(llm_response)

            if hasattr(llm_response, 'tool_calls') and len(llm_response.tool_calls) > 0:  # FIXME : should be done without hasattr ideally (using some AIMessage guaranteed attribute)
                for tool_call in llm_response.tool_calls:
                    tool_name = tool_call["name"].lower()
                    yield ToolCallInitialization(tool_name=tool_name, args=tool_call["args"])
                    
                    try:
                        if tool_name in self.interruptions: 
                            # Raise an exception with the tool data to interrupt execution and handle the result programatically.
                            if sync_mode:
                                tool_result = self.tools[tool_name].run(tool_call)
                            else:
                                tool_result = await self.tools[tool_name].arun(tool_call)
                            raise ToolInterrupt(tool_name, tool_result=tool_result.content)
                        
                        elif n_tool_called >= self.max_tool_depth:
                            yield ToolCallError(content=f"Exceeded maximum number of tools to be called in a row ({self.max_tool_depth})")
                            keep_streaming = False
                            break

                        else:
                            if sync_mode:
                                for event in self.tools[tool_name].stream(tool_call):
                                    self._update_messages(messages=messages, tool_call_id=tool_call["id"], event=event)
                                    yield event
                            else:
                                async for event in self.tools[tool_name].astream(tool_call):
                                    self._update_messages(messages=messages, tool_call_id=tool_call["id"], event=event)
                                    yield event
                            # Update the messages with temporary tool messages so the result is handled by the LLM.
                            n_tool_called += 1

                    except ToolInterrupt as e:
                        raise e
                    except Exception as e:
                        if n_consecutive_errors >= self.max_retries:
                            # Create a special error message for exceptions that are unrelated to the tool execution.
                            # NOTE : Exceptions related to the execution of the tool itself are handled using ToolCallError objects, handled in the loop above.
                            yield ToolCallError(content=self._create_error_tracking_message(tool_call, str(e)))
                        else:
                            n_consecutive_errors += 1
            else:
                yield llm_response
                keep_streaming = False
            
    def stream(self,
        input: LanguageModelInput, 
        config: Optional[RunnableConfig] = None, 
        stream_text: bool = True,
        **kwargs
    ) -> Generator[BaseMessage|StreamEvent, StreamFlags, None]:
        """
        Stream the llm output as well as special events when tools are called.

        stream_text : whether to yield chunks of llm response obtained via stream or to yield entire str responses.
        NOTE : If stream_text is False then the streaming will yield full text messages (str) and StreamEvents.
        If it is set to True, the streaming will yield text message chunks (str) and StreamEvents.
        NOTE : stream_text can only be set to True if the LLM supports the .stream or .astream methods.
        """
        async_gen_instance = self.astream(input=input, config=config, stream_text=stream_text, sync_mode=True, **kwargs)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                while True:
                    yield run_in_parallel_event_loop(future=async_gen_instance.__anext__())
            else:
                while True:
                    yield loop.run_until_complete(future=async_gen_instance.__anext__())
        except StopAsyncIteration:
            pass

    async def ainvoke(self, 
        input: LanguageModelInput, 
        config: Optional[RunnableConfig] = None, 
        sync_mode: bool = False,
        **kwargs
    ) -> BaseMessage:
        """
        Run a LLM and handle tool calls automatically until a final text response with no more tool call is received.
        Return an AIMessage containing the last textual response from the LLM.
        Return a SystemMessage containing the last error that's occurred in case of an error.
        """
        lastest_exception: Optional[ToolCallError] = None
        latest_response: Optional[AIMessage] = None
        async for event in self.astream(input=input, config=config, sync_mode=sync_mode, stream_text=False, **kwargs):
            if isinstance(event, ToolCallError):
                # Store the exception for returning in case of no result is generated
                lastest_exception = event
            elif isinstance(event, StreamEvent):
                pass  # Let the tools run themselves
            elif isinstance(event, AIMessage):
                # First test event received contains the LLM response
                if latest_response:
                    latest_response.content += "\n\n" + event.content  # XXX : Merging AIMessage
                else:
                    latest_response = event
            else:
                raise ValueError(f"Unexpected event type: {type(event)}")
        
        if latest_response:
            return latest_response
        elif lastest_exception:
            return SystemMessage(content=lastest_exception.content)
        else:
            raise ValueError("Streaming produced no output, which is abnormal.")

    def invoke(self, 
        input: LanguageModelInput, 
        config: Optional[RunnableConfig] = None, 
        **kwargs
    ) -> BaseMessage:
        """
        Run a LLM and handle tool calls automatically until a final text response with no more tool call is received.
        Return an AIMessage containing the last textual response from the LLM.
        Return a SystemMessage containing the last error that's occurred in case of an error.
        """
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return run_in_parallel_event_loop(future=self.ainvoke(input=input, config=config, sync_mode=True, **kwargs))
        else:
            return asyncio.run(main=self.ainvoke(input=input, config=config, sync_mode=True, **kwargs))
