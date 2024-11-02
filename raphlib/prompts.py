"""
Provide additional utility functions for managing a LLM conversation prompt.
This is designed to work on top of langchain's current capabilities.
"""
from pydantic import BaseModel
from typing import (
    Any, 
    List, 
    Tuple, 
    Union, 
    Dict, 
    Type, 
    Literal,
    Optional,
)
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.prompts.chat import ChatPromptTemplate, MessageLikeRepresentation
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.prompt_values import PromptValue

from .helpers import escape_characters

MessageLike = Union[str, Union[Tuple[str, str], Tuple[Literal["human", "ai", "system"], str, str]]]
MessageInput = Union[MessageLike, List[MessageLike]]


class ChatMessage(BaseModel):
    """
    A simple model for representing messages internally with chat history.
    """
    type: str
    content: str

    def create_copy(self, type: str):
        return ChatMessage(type=type, content=self.content)


class ChatHistory(BaseModel, Runnable):  # TODO : Make it serializable, based on the {'messages': List[dict], 'types': Dict} structure.
    """
    A class that handles messages and their types, dynamically creating a ChatPromptTemplate when needed.
    """
    name: Optional[str] = "ChatHistory"  # Comes from the Runnable class. The Runnable class handles the case where name is None.
    messages: List[ChatMessage] = []
    types: Dict[str, Literal["HumanMessage", "SystemMessage", "AIMessage", "ToolMessage"]] = {"system": "SystemMessage", "human": "HumanMessage", "ai": "AIMessage", "tool": "ToolMessage"}


    def size(self):
        return len(self.messages)


    def _select_messages(self, start_index: int, end_index: int) -> List[ChatMessage]:
        # Validate interval inputs
        if start_index is not None and (start_index < 0 or start_index >= len(self.messages)):
            raise IndexError("start_index is out of bounds.")
        if end_index is not None and (end_index > len(self.messages)):
            raise IndexError("end_index is out of bounds.")
        if start_index is not None and end_index is not None and start_index >= end_index:
            raise ValueError("start_index must be less than end_index.")

        return self.messages[start_index:end_index]


    def _to_base_message(self, messages: List[Any]) -> List[BaseMessage]:
        """
        Convert a message type to one of the built in names (system, human, ai) in the registry.
        """
        message_like_s: List[Tuple[str, str]] = [self._to_message_like(msg) for msg in messages]
        
        base_messages: List[BaseMessage] = []

        for (type, content) in message_like_s:

            if type == "system":
                base_messages.append(SystemMessage(content=content))
            elif type == "human":
                base_messages.append(HumanMessage(content=content))
            elif type == "ai":
                base_messages.append(AIMessage(content=content))
            else:
                raise ValueError(f"Unknown message BaseMessage subclass '{type}' for {content}.")
            
        return base_messages
        

    def _to_message_like(self, message: ChatMessage) -> Tuple[str, str]:
        """
        Convert a message to a tuple of parameters for ChatPromptTemplate.
        """
        langchain_type = self.types.get(message.type)
        if langchain_type == "SystemMessage":
            return ("system", message.content)
        elif langchain_type == "ToolMessage":
            return ("system", message.content)
        elif langchain_type == "AIMessage":
            return ("ai", message.content)
        elif langchain_type == "HumanMessage":
            if message.type == "human":
                return ("human", message.content)
            else:
                return ("human", f"{message.type}: {message.content}")  # Add type prefix
        else:
            raise ValueError(f"Unknown message BaseMessage subclass '{langchain_type}' for {message}.")


    def _convert_to_messages(self, messages: MessageInput, message_part: str = None, third_message_part: str = None, keep_variables = False) -> List[ChatMessage]:
        """
        Create compatible messages as {type=str, content=str} from MessageInput.
        
        Allowed input formats:
        - 'langchain_type', 'type', 'content'
        - 'type', 'content'
        - A string 'content'
        - A list of tuples of parameters, e.g., [ ('langchain_type', 'type', 'content'), ... ] 

        All inputs must be of type `str` or `List[Tuple]` containing the specified arguments.
        """
        if isinstance(messages, str):

            if third_message_part is not None and message_part is not None:  # Create a langchain type as specified.
                langchain_type = messages
                type = message_part
                content = third_message_part
                if not type in self.types.keys():
                    self.create_type(type, langchain_type)

            elif message_part is not None:  # Create the type as a human type if langchain type was not specified. (= speaker annotation)
                type = messages
                content = message_part
                if not type in self.types.keys():
                    self.create_type(type, "HumanMessage")

            else:  # Create a human type if nothing was specified. (= no speaker annotation on the message)
                type = "human"
                content = messages
                if not "human" in self.types.keys():
                    self.create_type("human", "HumanMessage")

            return [
                ChatMessage(
                    type = type, 
                    content = content if keep_variables else escape_characters(content)
                )
            ]

        elif isinstance(messages, list):
            converted_messages = []
            for msg in messages:
                if isinstance(msg, tuple):
                    converted_messages.extend(self._convert_to_messages(*msg, keep_variables=keep_variables))

                elif isinstance(msg, (str, BaseMessage)):
                    converted_messages.extend(self._convert_to_messages(msg, keep_variables=keep_variables))

                elif isinstance(msg, ChatMessage):  # Append chat messages directly
                    converted_messages.append(msg)

                else:
                    raise ValueError(f"Invalid message format in the message list {messages}. Expected string or tuple.")
            return converted_messages
        
        elif isinstance(messages, BaseMessage):

            content = messages.content if keep_variables else escape_characters(messages.content)

            if isinstance(messages, SystemMessage):
                return [ChatMessage(type="system", content=content)]
            elif isinstance(messages, AIMessage):
                return [ChatMessage(type="ai", content=content)]
            elif isinstance(messages, ToolMessage):
                return [ChatMessage(type="system", content=content)]
            elif isinstance(messages, HumanMessage) or isinstance(messages, BaseMessage):
                return [ChatMessage(type="human", content=content)]
            else:
                raise ValueError(f"Unknown message BaseMessage subclass '{message.__class__.__name__}'.")  # Useless until isinstance(messages, BaseMessage) is removed
            
        elif isinstance(messages, tuple):
            return self._convert_to_messages(*messages, keep_variables=keep_variables)
        
        elif isinstance(messages, ChatMessage):
            return [messages]
        
        raise ValueError("Invalid message format. Expected string, tuple, or a list. Got neither.")
        

    def create_type(self, label: str, langchain_type: Literal["HumanMessage", "SystemMessage", "AIMessage", "ToolMessage"]) -> None:
        """
        Create a new message type.
        """
        if not langchain_type in ("SystemMessage", "HumanMessage", "AIMessage", "ToolMessage"):
            raise ValueError(f"Unknown langchain type '{langchain_type}'.")
        if label == "ai" and langchain_type != "AIMessage":
            self.types[label] = "AIMessage"
            print(f"The 'ai' has been forced into a AIMessage type instead of a {langchain_type} to conserve default types homogeneity.")
        elif label == "system" and langchain_type != "SystemMessage":
            self.types[label] = "SystemMessage"
            print(f"The'system' has been forced into a SystemMessage type instead of a {langchain_type} to conserve default types homogeneity.")
        self.types[label] = langchain_type


    def copy(self) -> 'ChatHistory':
        """
        Return a deep copy of the ChatHistory object.
        """
        return ChatHistory(
            messages = self.messages.copy(),
            types = self.types.copy(),
        )


    def remove(
        self,
        types_to_remove: List[str]
    ) -> 'ChatHistory':
        """
        Remove messages of the specified types in-place.
        """
        self.messages = [
            msg for msg in self.messages 
            if msg.type not in types_to_remove
        ]
        return self

    def restrict(
        self,
        types_to_keep: List[str]
    ) -> 'ChatHistory':
        """
        Keep the messages of the specified types in-place.
        """
        self.messages = [
            msg for msg in self.messages 
            if msg.type in types_to_keep
        ]
        return self
    
    def replace(self, types_replacement_map: Dict[str, str]) -> 'ChatHistory':
        """
        Replace messages of a specific type with another type.
        Modifies the original object and raises an error if the type or replacement type doesn't exist in the type registry.
        
        Example: chat.replace({"human": "system"}).
        """
        for old_type, new_type in types_replacement_map.items():
            if old_type not in self.types:
                raise ValueError(f"Original type '{old_type}' does not exist in the type registry.")
            
            if new_type not in self.types:
                old_type_langchain = self.types[old_type]
                print(f"Replacement type '{new_type}' does not exist in the type registry. Creating a new type as {old_type_langchain}.")  # TODO : Make it a log
                self.create_type(new_type, old_type_langchain)
                
            self.messages = [
                msg.create_copy(type=new_type) if msg.type == old_type else msg
                for msg in self.messages
            ]
        return self
    

    def without(
        self,
        types_to_remove: List[str],
    ) -> 'ChatHistory':
        """
        Return a copy without the specified message types.
        """
        return self.copy().remove(types_to_remove)

    def only(
        self,
        types_to_keep: List[str]
    ) -> 'ChatHistory':
        """
        Return a copy with only the specified message types.
        """
        return self.copy().restrict(types_to_keep)

    def using(self, types_replacement_map: Dict[str, str]) -> 'ChatHistory':
        """
        Returns a copy of the object where messages of a specific type are replaced by another type.
        Raises an error if the type or replacement type doesn't exist in the type registry.
        
        Example: chat.using({"human": "system"}).
        """        
        return self.copy().replace(types_replacement_map)
    
    def using_prompt(self, *args, keep_variables: bool = False) -> 'ChatHistory':
        """
        Return a copy of the object with the additional prompt inserted on the left.
        """
        return self.copy().insert(0, *args, keep_variables=keep_variables)
    
    def using_end_prompt(self, *args, keep_variables: bool = False) -> 'ChatHistory':
        """
        Return a copy of the object with the additional prompt inserted on the right.
        """
        return self.copy().append(*args, keep_variables=keep_variables)


    def append(
        self,
        *args,
        keep_variables: bool = False,
    ) -> 'ChatHistory':
        """
        Append messages to the current list.

        Allowed input formats:
        - 'langchain_type', 'type', 'content'
        - 'type', 'content'
        - A string 'content'
        - A list of tuples of parameters, e.g., [ ('langchain_type', 'type', 'content'), ... ] 
        """
        self.messages.extend(self._convert_to_messages(*args, keep_variables=keep_variables))
        return self

    def insert(
        self,
        index: int,
        *args,
        keep_variables: bool = False,
    ) -> 'ChatHistory':
        """
        Insert a message or a list of messages at the given index.

        Allowed input formats:
        - 'langchain_type', 'type', 'content'
        - 'type', 'content'
        - A string 'content'
        - A list of tuples of parameters, e.g., [ ('langchain_type', 'type', 'content'), ... ] 
        """
        for msg in reversed(self._convert_to_messages(*args, keep_variables=keep_variables)):
            self.messages.insert(index, msg)
        return self
    
    def appendleft(self, *args, keep_variables: bool = False) -> 'ChatHistory':
        """
        Insert the input at position 0.

        Allowed input formats:
        - 'langchain_type', 'type', 'content'
        - 'type', 'content'
        - A string 'content'
        - A list of tuples of parameters, e.g., [ ('langchain_type', 'type', 'content'), ... ] 
        """
        return self.insert(0, *args, keep_variables=keep_variables)

    def pop(
        self,
        index: int = -1
    ) -> 'ChatHistory':
        """
        Remove and return the message at the given index (last message by default).
        """
        self.messages.pop(index)
        return self
    
    def delete_interval(self, start_index: int, end_index: int) -> 'ChatHistory':
        """
        Remove messages from the specified interval [start_index, end_index).
        """
        if start_index < 0 or end_index > len(self.messages) or start_index >= end_index:
            raise IndexError("Invalid interval specified for deletion.")

        del self.messages[start_index:end_index]
        return self

    def overwrite(
        self,
        messages: MessageInput,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        ignore_type: Optional[List[str]] = None,
        keep_variables: bool = False,
    ) -> 'ChatHistory':
        """
        Overwrite messages in the specified interval [start_index, end_index).
        If no interval is provided, the entire message list is replaced.

        With `ignore_type`, only messages not in `ignore_type` within the interval are overwritten.

        - Keeps track of the position of the first message whose type is not in `ignore_type`.
        - Deletes all messages that are not in `ignore_type` within the interval.
        - Inserts the new messages at the position of the first non-ignored message.

        Allowed input formats to replace messages:
        - A tuple ('langchain_type', 'type', 'content')
        - A tuple ('type', 'content')
        - A string 'content'
        - A list of tuples of the above formats, e.g.,
          [ ('langchain_type', 'type', 'content'), ... ]
        """
        # Convert the provided messages to a list of ChatMessage objects
        converted_messages = self._convert_to_messages(messages, keep_variables=keep_variables)

        # Handle case where interval is not provided (replace entire list)
        if start_index is None and end_index is None:
            if ignore_type:
                # Find the first message not in ignore_type
                first_non_ignore = next(
                    (i for i, msg in enumerate(self.messages) if msg.type not in ignore_type),
                    None
                )
                if first_non_ignore is not None:
                    # Delete all non-ignore_type messages in the entire list
                    self.messages = [
                        msg for msg in self.messages if msg.type in ignore_type
                    ]
                    # Insert the new messages at the first_non_ignore position
                    if first_non_ignore > len(self.messages):
                        first_non_ignore = len(self.messages)
                    self.messages.insert(first_non_ignore, *converted_messages)
                else:
                    # If all messages are to be ignored, append the new messages at the end
                    self.messages.extend(converted_messages)
            else:
                self.messages = converted_messages
            return self

        # Validate interval inputs
        if start_index is None or end_index is None:
            raise ValueError("Both start_index and end_index must be provided for interval replacement.")
        if start_index < 0 or end_index > len(self.messages) or start_index >= end_index:
            raise IndexError("Invalid interval specified for overwriting.")

        if not ignore_type:
            # Perform standard overwrite
            self.delete_interval(start_index, end_index)
            self.messages[start_index:start_index] = converted_messages
        else:
            # Find the first message in the interval not in ignore_type
            first_non_ignore = None
            for i in range(start_index, end_index):
                if self.messages[i].type not in ignore_type:
                    first_non_ignore = i
                    break

            if first_non_ignore is not None:
                # Delete all messages in [start_index, end_index) not in ignore_type
                self.messages = [
                    msg for idx, msg in enumerate(self.messages)
                    if not (start_index <= idx < end_index and msg.type not in ignore_type)
                ]
                # Adjust end_index after deletion
                new_end_index = first_non_ignore
                # Insert the new messages at the position of the first non-ignored message
                self.messages.insert(first_non_ignore, *converted_messages)
            else:
                # If all messages in the interval are to be ignored, do not perform overwrite
                pass  # Alternatively, you could choose to append or raise an exception

        return self
    

    def as_base_message(self, start_index: Optional[int] = None, end_index: Optional[int] = None) -> List[BaseMessage]:
        """
        Retrieve messages as BaseMessage instances, 
        in the specified interval [start_index, end_index].
        If no interval is provided, return the entire list of messages.
        """
        if start_index is None and end_index is None:
            return self._to_base_message(self.messages) # Return all messages if no interval is specified
        else:
            return self._to_base_message(
                self._select_messages(start_index, end_index)
            )

    
    def as_message_like(self, start_index: Optional[int] = None, end_index: Optional[int] = None) -> Tuple[str, str]:
        """
        Retrieve messages as a list of (langchain type, content),
        in the specified interval [start_index, end_index).
        If no interval is provided, return the entire list of messages.
        """
        if start_index is None and end_index is None:
            messages = self.messages # Return all messages if no interval is specified
        else:
            messages = self._select_messages(start_index, end_index)

        return [self._to_message_like(msg) for msg in messages]

    def invoke(self, input: dict, config: Optional[RunnableConfig] = None, **kwargs) -> PromptValue:
        return ChatPromptTemplate(self.as_message_like()).invoke(input, config, **kwargs)
    
    def pretty(self, input: dict = {}, config: Optional[RunnableConfig] = None, **kwargs) -> str:
        s = "ChatHistory ========================\n\n"
        messages = self.invoke(input, config, **kwargs).messages
        for message in messages:
            s += message.__class__.__name__
            s += " \t>\n"
            s += message.content + "\n\n"
        s += "=====================================\n"
        return s
    
    def pretty_print(self, input: dict = {}, config: Optional[RunnableConfig] = None, **kwargs):
        print(self.pretty(input, config, **kwargs))







# Example Usage
if __name__ == "__main__":
    message = ChatMessage(type="system", content="Welcome to the chat history!")
    chat = ChatHistory(messages=[message])

    jsond = chat.model_dump_json()
    print(jsond, type(jsond))

    chat2 = ChatHistory.model_validate_json(jsond)
    print(chat2.model_dump())

    print("===== ChatHistory Testing =====\n")

    # 1. Create initial messages.
    message1 = ChatMessage(type="system", content="System initialized.")
    message2 = ChatMessage(type="human", content="Hello, how are you?")
    message3 = ChatMessage(type="ai", content="I'm doing well, thank you for asking!")
    
    # 2. Initialize ChatHistory and append initial messages.
    chat = ChatHistory(messages=[message1, message2, message3])
    print("\n==== Initial ChatHistory ====")
    print(chat)

    # 3. Append new messages using different formats.
    chat.append('AIMessage', "assistant", "I'm an AI assistant.")
    chat.append("Hello world!")  # Defaults to human message
    print("\n==== ChatHistory after append ====")
    print(chat)

    # 4. Insert new messages at a specific position.
    chat.insert(1, 'system', "Reminding you about our policies")
    print("\n==== ChatHistory after insert ====")
    print(chat)

    # 5. Remove a message (pop) from the end and display the history.
    chat.pop()
    print("\n==== ChatHistory after pop ====")
    print(chat)

    # 6. Replace a message type in the entire history.
    chat.replace({"human": "ai"})
    print("\n==== ChatHistory after replace (human -> ai) ====")
    print(chat)

    # 7. Restrict the history to only AI messages.
    chat.restrict(["ai"])
    print("\n==== ChatHistory after restrict (only ai) ====")
    print(chat)

    # 8. Add new types and modify.
    chat.create_type("agent", "ToolMessage")
    chat.create_type("operator", "HumanMessage")
    chat.append("agent", "These are tool-based test messages.")
    chat.append("operator", "This is a message from the operator.")
    print("\n==== ChatHistory after adding new types (agent, operator) ====")
    print(chat)

    # 9. Test the without() by removing 'system' type messages.
    chat.without(['system'])
    print("\n==== ChatHistory after without ('system') ====")
    print(chat)

    # 10. Test overwrite function by replacing a range of messages.
    chat.overwrite(("ai", "Hello again!"), 0, 2)
    print("\n==== ChatHistory after overwrite (first 2 messages) ====")
    print(chat)

    # 11. Display chat history as base messages (convert to LangChain format).
    base_messages = chat.as_base_message()
    print("\n==== ChatHistory as base messages (LangChain format) ====")
    for msg in base_messages:
        print(f"Type: {msg.__class__.__name__}, Content: {msg.content}")

    # 12. Test deep copy.
    chat_copy = chat.copy()
    chat_copy.append("system", "This is a system message in the copied chat.")
    print("\n==== Original ChatHistory ====")
    print(chat)
    print("\n==== Copied ChatHistory after modifications ====")
    print(chat_copy)

    # 13. Test history dump and restore using JSON.
    json_dump = chat.model_dump_json()
    print("\n==== ChatHistory JSON Dump ====")
    print(json_dump)

    chat_restored = ChatHistory.model_validate_json(json_dump)
    print("\n==== Restored ChatHistory from JSON ====")
    print(chat_restored)

    # 14. Interval-based delete and overwrite.
    chat.delete_interval(1, 3)
    print("\n==== ChatHistory after deleting messages from index 1 to 3 ====")
    print(chat)

    chat.overwrite([('AIMessage', 'blahtor', 'This is an overwrite test')], 0, 1)
    print("\n==== ChatHistory after overwriting index 0 ====")
    print(chat)

    # 15. Invoke ChatPromptTemplate with a specific input.
    chat3 = ChatHistory()
    chat3.append([
        ('system', 'Hello {name}. Please perform the following task : {task}'), 
        ('ai', 'Hello {name}. No.'), 
        ('john', 'Please, if you do this I will give you {gift}.')
    ])
    print(chat3.invoke({'name': 'John', 'task': 'complete the task', 'gift': 'a gift'}))
