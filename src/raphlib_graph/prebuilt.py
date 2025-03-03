
from typing import (
    List,
    Optional,
    TypeVar,
    Generic,
)
from pydantic import BaseModel
from langchain.chat_models.base import BaseChatModel

from ..raphlib import ChatHistory, LLMFunction, LLMWithTools, LangchainMessageTypes, BaseTool

from .base import (
    BaseGraph, 
    BaseState,
)


class ChatState(BaseState):
    history: ChatHistory = ChatHistory()

State = TypeVar("State", bound=ChatState)

class Graph(BaseGraph[State], Generic[State]):

    def redirect_node(
            self, 
            name: str, 
            llm: BaseChatModel, 
            prompt: str, 
            types: Optional[List[LangchainMessageTypes]] = None
        ):
        """
        Create a node in the graph. It will run prompt with the current chat appended to **select the next node in the graph**.

        Parameters:
            name (str): The name of the node.
            llm (BaseChatModel): The model to call with the prompt.
            prompt (str): A prompt for **finding the name of the next node**. The current chat will be appended at the end of this prompt.
            types (List[str] | None): The chat message types to show below the prompt. If unspecified, show all messages.
        """
        pass


    def chat_node(self, name: str, tools: List[BaseTool] = []):
        """
        Create a node on the graph that runs a basic chat.

        Parameters:
            name (str): The name of the node.
            tools (List[BaseTool]): The (Raphlib) tools the llm will have access to.
        
        Notes:
            If a ToolInterrupt is triggered, the graph will jump 
            to the node whose name is in the tool_result field.
        """
        pass

