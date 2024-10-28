from typing import (
    Any, 
    List, 
    Optional, 
)

from langchain_core.tools import StructuredTool
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from .helpers import escape_characters


class ToolInterrupt(Exception):
    def __init__(self, tool_name, tool_result, *args):
        self.tool_name = tool_name
        self.tool_result = tool_result
        super().__init__(f"Event {self.tool_name} was triggered. Value is {self.tool_result}.", *args)


class LLMWithTools(Runnable[LanguageModelInput, BaseMessage]):
    """
    A non serializable runnable that behaves like a BaseChatModel when invoked, but manages tools automatically.
    """
    def __init__(self, llm: BaseChatModel, tools: List[StructuredTool], tool_events: List[str] = [], track_call: bool = False):
        """
        This class behaves like a BaseChatModel except for the fact that it 
        automatically sends tool responses to llms that require tool calls, 
        and returns only AIMessages that do not contain tool calls anymore.

        track_call: bool -> If True tool call parameters and results are added to the conversation as system messages.
        If this option is set to True, the langchain chain has to support a List[BaseMessage] input instead of a single AIMessage object.
        """
        self.tools = {tool.name: tool for tool in tools}
        self.tool_events = tool_events
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
            if hasattr(llm_response, 'tool_calls') and len(llm_response.tool_calls) > 0:
                for tool_call in llm_response.tool_calls:
                    tool_name = tool_call["name"].lower()
                    tool_result = self.tools[tool_name].invoke(tool_call)  # Invoke with the whole tool call object
                    if tool_name in self.tool_events: 
                        # Raise an exception with the tool data to interrupt execution and handle the result programatically.
                        raise ToolInterrupt(tool_name, tool_result)
                    else:
                        # Update the messages with temporary tool messages so the result is handled by the LLM.
                        messages.append(ToolMessage(tool_result, tool_call_id=tool_call["id"]))

                        if track_call:  # Update the tracking history
                            call_tracking.append(self._create_tracking_message(tool_call, tool_result))
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
        ... # invoke but all the invoke are replaced by ainvoke