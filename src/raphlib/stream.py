import logging
import asyncio

from typing import (
    Any, 
    List, 
    Dict,
    Type,
    Union, 
    Optional, 
    Generator,
    AsyncGenerator,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel

from langchain_core.prompt_values import ChatPromptValue
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, ToolMessage, AIMessageChunk
from langchain_core.messages.ai import add_ai_message_chunks

from .helpers import escape_characters, run_in_parallel_event_loop, get_or_create_event_loop
from .events import *
from .tools_old import BaseTool


class LLMWithTools(Runnable[LanguageModelInput, BaseMessage]):
    """
    A non serializable runnable that behaves like a BaseChatModel when invoked, but manages tools automatically.
    """
    def __init__(
            self, 
            llm: BaseChatModel, 
            tools: List[Union[BaseTool,Type[BaseTool]]], 
            interrupt_message_type: str = "system", 
            max_retries: int = 2,
            max_tool_depth: int = 10,
        ):
        """
        Initialize an LLMWithTools instance that behaves like a BaseChatModel with automatic tool management.

        Parameters:
            llm (BaseChatModel): The underlying language model.
            tools (List[Union[BaseTool, Type[BaseTool]]]): A list of tools or tool classes to integrate.
                Uninitialized tools will be instantiated.
            max_retries (int, optional): Maximum retry attempts for tool invocations (default is 2).
            max_tool_depth (int, optional): Maximum allowed depth for nested tool calls (default is 10).

        Note:
            Make sure you handle ToolInterrupt exceptions if any of your tools uses them.
        """
        tools = [tool() if isinstance(tool, type) else tool for tool in tools]  # Initialize any uninitialized tool

        self.tools = {tool.name: tool for tool in tools}
        if self.tools:
            self.llm: BaseChatModel = llm.bind_tools(tools)
        else:
            self.llm: BaseChatModel = llm
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

    def add_tools(self, tools: List[BaseTool] = [], interruptions: List[str] = []):
        """
        Add the provided tools to the list.
        If a tool with a similar name already exists replace it with the provided one.
        """
        self.tools.update({tool.name: tool for tool in tools})
        self.interruptions.extend(interruptions)
        self.llm: BaseChatModel = self.llm.bind_tools(tools)

    def overwrite_tools(self, tools: List[List[BaseTool]] = [], interruptions: List[str] = []):
        """
        Overwrite the tools and iterruptions with the provided lists.
        If no parameters are passed, simply reset all the tools.
        """
        self.tools = {tool.name: tool for tool in tools}
        self.interruptions = interruptions

    async def astream(self,
        input: LanguageModelInput | ToolInterrupt, 
        config: Optional[RunnableConfig] = None, 
        sync_mode: bool = False,
        stream_text: bool = True,
        **kwargs
    ) -> AsyncGenerator[AIMessageChunk|StreamEvent|AIMessage, StreamFlags]:
        """
        This generator runs the asynchronous streaming process (implemented in astream) within the current event loop or spins up a new one using
        an appropriate runner. It yields data as soon as they become available, using the same streaming protocol as astream.

        The protocol supports the following event types:
        * AIMessageChunk: When stream_text is True, these are incremental text outputs from the language model.
        * AIMessage: When stream_text is False, complete output messages are yielded.
        * StreamEvent: Additional events, such as ToolCallInitialization and ToolCallError, are yielded when tools are invoked or errors occur.

        Parameters:
            input (LanguageModelInput, ToolInterrupt):
                The input data for the language model, which is internally transformed for processing. 
                A ToolInterrupt can also be provided as an input to resume a halted tool calling process.
            config (Optional[RunnableConfig], optional):
                Configuration parameters to fine-tune the model's behavior.
            stream_text (bool, optional):
                Specifies whether to yield incremental text chunks (True) or complete responses (False).
            **kwargs:
                Any extra keyword arguments to be passed on to the asynchronous streaming method.

        Yields:
            Generator Events (Union[AIMessageChunk, StreamEvent, AIMessage]):
                The synchronous generator yields events from the streaming process, each associated with StreamFlags for additional status.

        Raises:
            ToolInterrupt:
                When they are triggered by nodes, after populating them with tool call information.

        Note:
            - This method wraps the asynchronous astream method. When the underlying event loop is already running, it uses a helper 
                (run_in_parallel_event_loop) to ensure compatibility and correctness.
            - If no running loop is detected, the helper will start one and consume all asynchronous events until completion.
            - As with astream, ensure that the language model supports streaming operations suitable for the chosen mode.
        """
        if isinstance(input, ToolInterrupt):
            resume = input
            messages = resume.messages
            n_tool_called = resume.n_tool_called
        else:
            resume = None
            prompt: ChatPromptValue = self.llm._convert_input(input)
            messages: List[BaseMessage] = prompt.to_messages()  # Extract the messages from the prompt so it's compatible with llm invoke.
            n_tool_called: int = 0

        keep_streaming: bool = True
        while keep_streaming:

            n_consecutive_errors: int = 0

            if resume:  # Skips the first generation of an output

                llm_response = resume.llm_response
                stop_processing_tools_flag = False

            else:

                if sync_mode:
                    if stream_text:
                        # Streaming llm output unless it's a ToolCall
                        chunks: List[AIMessageChunk] = []
                        for chunk in self.llm.stream(messages, config, **kwargs):
                            chunks.append(chunk)
                            yield chunk
                        combined_chunk = add_ai_message_chunks(chunks[0], *chunks[1:]) if len(chunks) > 1 else chunks[0]
                        llm_response = AIMessage(**combined_chunk.model_dump(exclude=["type"]))
                    else:
                        llm_response: AIMessage = self.llm.invoke(messages, config, **kwargs)
                else:
                    if stream_text:
                        chunks: List[AIMessageChunk] = []
                        async for chunk in self.llm.astream(messages, config, **kwargs):
                            chunks.append(chunk)
                            yield chunk
                        combined_chunk = add_ai_message_chunks(chunks[0], *chunks[1:]) if len(chunks) > 1 else chunks[0]
                        llm_response = AIMessage(**combined_chunk.model_dump(exclude=["type"]))
                    else:
                        llm_response: AIMessage = await self.llm.ainvoke(messages, config, **kwargs)

                messages.append(llm_response)
                stop_processing_tools_flag = False  # True when encountering a stopping condition

            if hasattr(llm_response, 'tool_calls') and len(llm_response.tool_calls) > 0:
                
                for i, tool_call in enumerate(llm_response.tool_calls):  # NOTE : All tools must be processed and provided a response, even if it's an error.
                    
                    if resume:
                        if i < resume.current_tool_index:
                            continue  # Skips any index that was already processed
                        elif i == resume.current_tool_index:
                            resume = None  # Delete the resume variable so that the next tools are processed properly
                            continue
                    
                    tool_name = tool_call["name"].lower()
                    yield ToolCallInitialization(tool_name=tool_name, args=tool_call["args"])
                    
                    try:
                        if stop_processing_tools_flag:
                            messages.append(ToolMessage(content="Tool skipped because of an error with the previous tool.", tool_call_id=tool_call["id"]))
                        
                        elif n_tool_called >= self.max_tool_depth:  # INTERRUPTION BECAUSE OF RECURSION DEPTH
                            yield ToolCallError(content=f"Exceeded maximum number of tools to be called in a row ({self.max_tool_depth})", tool_name=tool_name)
                            keep_streaming = False
                            stop_processing_tools_flag = True

                        else:  # REGULAR TOOL CALLING
                            if sync_mode:
                                for event in self.tools[tool_name].stream(tool_call):
                                    yield event
                                messages.append(ToolMessage(content=event.content, tool_call_id=tool_call["id"]))  # Last Event is a tool result or error
                            else:
                                async for event in self.tools[tool_name].astream(tool_call):
                                    yield event
                                messages.append(ToolMessage(content=event.content, tool_call_id=tool_call["id"]))  # Last Event is a tool result or error
                            # Update the messages with temporary tool messages so the result is handled by the LLM.
                            n_tool_called += 1

                    except ToolInterrupt as interrupt:

                        messages.append(ToolMessage(content=interrupt.tool_result, tool_call_id=tool_call["id"]))
                        yield ToolCallResult(content=interrupt.tool_result, tool_name=tool_name)

                        interrupt.add_stream_metadata(
                            tool_name = tool_name,
                            messages = messages,
                            llm_response = llm_response,
                            current_tool_index = i,
                            n_tool_called = n_tool_called + 1,
                        )
                        raise interrupt
                    
                    except BaseInterrupt as interrupt:

                        raise interrupt

                    except Exception as e:

                        if n_consecutive_errors >= self.max_retries:
                            # NOTE : Exceptions related to the execution of the tool itself are handled using ToolCallError objects, in the loop above.
                            yield ToolCallError(content=self._create_error_tracking_message(tool_call, f"{type(e)} {str(e)}"), tool_name=tool_name)

                            messages.append(ToolMessage(content=f"An error occurred while running the tool : {type(e)} {str(e)}", tool_call_id=tool_call["id"]))
                            stop_processing_tools_flag = True
                        else:
                            n_consecutive_errors += 1
                            print(f"An error occurred while running the tool : {type(e)} {str(e)}")
                            messages.append(ToolMessage(content=f"An error occurred while running the tool : {type(e)} {str(e)}", tool_call_id=tool_call["id"]))

            else:
                yield llm_response
                keep_streaming = False
            
    def stream(self,
        input: LanguageModelInput | ToolInterrupt,  
        config: Optional[RunnableConfig] = None, 
        stream_text: bool = True,
        **kwargs
    ) -> Generator[AIMessageChunk|StreamEvent|AIMessage, StreamFlags, None]:
        """
        This generator runs the asynchronous streaming process (implemented in astream) within the current event loop or spins up a new one using
        an appropriate runner. It yields data as soon as they become available, using the same streaming protocol as astream.

        The protocol supports the following event types:
            * AIMessageChunk: When stream_text is True, these are incremental text outputs from the language model.
            * AIMessage: When stream_text is False, complete output messages are yielded.
            * StreamEvent: Additional events, such as ToolCallInitialization and ToolCallError, are yielded when tools are invoked or errors occur.

        Parameters:
            input (LanguageModelInput, ToolInterrupt):
                The input data for the language model, which is internally transformed for processing. 
                A ToolInterrupt can also be provided as an input to resume a halted tool calling process.
            config (Optional[RunnableConfig], optional):
                Configuration parameters to fine-tune the model's behavior.
            stream_text (bool, optional):
                Specifies whether to yield incremental text chunks (True) or complete responses (False).
            **kwargs:
                Any extra keyword arguments to be passed on to the asynchronous streaming method.

        Yields:
            Generator Events (Union[AIMessageChunk, StreamEvent, AIMessage]):
                The synchronous generator yields events from the streaming process, each associated with StreamFlags for additional status.

        Note:
            - This method wraps the asynchronous astream method. When the underlying event loop is already running, it uses a helper 
                (run_in_parallel_event_loop) to ensure compatibility and correctness.
            - If no running loop is detected, the helper will start one and consume all asynchronous events until completion.
            - As with astream, ensure that the language model supports streaming operations suitable for the chosen mode.
        """
        async_gen_instance = self.astream(input=input, config=config, stream_text=stream_text, sync_mode=True, **kwargs)
        try:

            loop = get_or_create_event_loop()
            if loop.is_running():
                while True:
                    yield run_in_parallel_event_loop(future=async_gen_instance.__anext__())
            else:
                while True:
                    yield loop.run_until_complete(future=async_gen_instance.__anext__())

        except BaseInterrupt as interrupt:
            raise interrupt
        
        except StopAsyncIteration:
            pass

    async def ainvoke(self, 
        input: LanguageModelInput, 
        config: Optional[RunnableConfig] = None, 
        sync_mode: bool = False,
        **kwargs
    ) -> AIMessage:
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
    ) -> AIMessage:
        """
        Run a LLM and handle tool calls automatically until a final text response with no more tool call is received.
        Return an AIMessage containing the last textual response from the LLM.
        Return a SystemMessage containing the last error that's occurred in case of an error.
        """
        loop = get_or_create_event_loop()
        if loop.is_running():
            return run_in_parallel_event_loop(future=self.ainvoke(input=input, config=config, sync_mode=True, **kwargs))
        else:
            return asyncio.run(main=self.ainvoke(input=input, config=config, sync_mode=True, **kwargs))
