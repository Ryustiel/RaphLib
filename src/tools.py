import logging
from typing import (
    Any, 
    List, 
    Union, 
    Optional, 
    Generator,
    AsyncGenerator,
)

from langchain_core.tools import StructuredTool
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel

from .helpers import escape_characters
from .stream import StreamEvent, StreamFlags, ToolCallError, ToolCallInitialization, ToolCallResult

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
    def __init__(self, llm: BaseChatModel, tools: List[StructuredTool], interruptions: List[str] = [], track_call: bool = False):
        """
        This class behaves like a BaseChatModel except for the fact that it 
        automatically sends tool responses to llms that require tool calls, 
        and returns the final AIMessage response, or a list of messages corresponding to the output.

        NOTE : If you are registering interruptions (len(self.interruptions) > 0), you have to handle ToolInterrupt exceptions.

        track_call: bool -> If True tool call parameters and results are added to the conversation as system messages.
        If this option is set to True, the langchain chain has to support a List[BaseMessage] input instead of a single AIMessage object.

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
        self.track_call = track_call

    def _create_tracking_message(self, tool_call, tool_result):
        """
        Create a system message with the tool call and result.
        """
        return SystemMessage(f"""
            Tool call: {tool_call['name']} {escape_characters(str(tool_call['args']))}\n
            Result: {escape_characters(str(tool_result.content))}
        """)
        
    def invoke(self, 
        input: LanguageModelInput, 
        config: Optional[RunnableConfig] = None, 
        track_call: bool = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        prompt: ChatPromptValue = self.llm._convert_input(input)
        messages: List[BaseMessage] = prompt.to_messages()  # Extract the messages from the prompt for easier manipulation.
        if track_call is None:
            track_call = self.track_call  # Use the provided track_call if it's not None, otherwise use the class default.
        if track_call:
            call_tracking: List[BaseMessage] = list()

        while True:
            llm_response: AIMessage = self.llm.invoke(messages, config, **kwargs)
            messages.append(llm_response)
            if hasattr(llm_response, 'tool_calls') and len(llm_response.tool_calls) > 0:  # FIXME : should be done without hasattr ideally (using some AIMessage guaranteed attribute)
                for tool_call in llm_response.tool_calls:
                    tool_name = tool_call["name"].lower()
                    
                    try:
                        tool_result = self.tools[tool_name].invoke(tool_call)  # Invoke with the whole tool call object
                        if tool_name in self.interruptions: 
                            # Raise an exception with the tool data to interrupt execution and handle the result programatically.
                            raise ToolInterrupt(tool_name, tool_result=tool_result)
                        else:
                            # Update the messages with temporary tool messages so the result is handled by the LLM.
                            messages.append(ToolMessage(tool_result, tool_call_id=tool_call["id"]))

                            if track_call:  # Update the tracking history
                                call_tracking.append(self._create_tracking_message(tool_call, tool_result))

                    except ToolInterrupt as e:
                        raise e
                    
                    except Exception as e:
                        logging.warning(f"Exception while calling tool {tool_name} : {e}")
                        messages.append(ToolMessage(content=f"Exception while calling tool {tool_name} : {e}", tool_call_id=tool_call["id"]))
            else:
                if track_call:
                    call_tracking.append(llm_response)
                    return call_tracking
                else:
                    return llm_response

    async def ainvoke(
        self, 
        input: LanguageModelInput, 
        config: Optional[RunnableConfig] = None, 
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        ... # invoke but all the tool invokes are run in the current loop whenever it's possible.
        # regular invoke should be a simple asyncio run of this function instead of its current state.
        # rethink the tool calling mechanism in here.

    def stream(
        self, 
        input: LanguageModelInput, 
        config: Optional[RunnableConfig] = None, 
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Generator[Union[str, StreamEvent], StreamFlags, None]:
        pass

    async def astream(
        self, 
        input: LanguageModelInput, 
        config: Optional[RunnableConfig] = None, 
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Union[str, StreamEvent], StreamFlags]:
        pass
