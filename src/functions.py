"""
Provide utility functions for calling LLMs and retrieve structured output.
"""
import logging
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
)
from pydantic import BaseModel
from pydantic_core import ValidationError

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, AIMessageChunk, AIMessage
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.prompt_values import PromptValue
from langchain_core.language_models.chat_models import BaseChatModel

from .helpers import escape_characters, run_in_parallel_event_loop, get_or_create_event_loop
from .tools import StreamEvent, ResetStream, StreamFlags
from .prompts import ChatHistory
from .stream import ToolInterrupt, LLMWithTools
from .stables import StableModel
from .parsers import OutputParser


class LLMFunctionResult:
    def __init__(self, success, result):
        self.success: bool = success
        self.result: Union[Exception, BaseModel] = result

class BatchLLMFunctionResult:
    def __init__(self, success, results):
        success: bool = success
        results: List[LLMFunctionResult] = results


class LLMFunction(Runnable):
    """
    An interface for creating LLM function calls with minimal effort.

    The prompt can be provided as a string (interpreted as a system prompt), a list of (role, message)
    tuples, or a ChatPromptTemplate. Keyword arguments are used to build a pydantic model (unless the
    special 'pydantic_model' argument is provided) that validates and parses the LLM's response.
    
    The class ensures that the LLM output conforms to a pydantic model, supports automatic retries
    on validation errors, and can return results either as a direct pydantic instance or as a tuple indicating success.
    
    ### Examples:
    
        # Function call using an external ChatPromptTemplate

        llmfunction = LLMFunction(llm,
            ( 
            "Note all the names mentioned in this conversation", 
            "as well as the number of questions that were asked.",
            )
            names=["raph", ...],
            number_of_questions=5,
        )

        conversation = ChatHistory([
            ("sam", "bye felicia."),
            ("sam", "{sentence}")
        ])

        (conversation | llmfunction).invoke({"sentence": "What about Henriette ?"})

        # The llm will be called on (
        #   "Note all the names mentioned in this conversation, 
        # as well as the number of questions that were asked."
        #   "sam: "bye felicia.",
        #   "sam: "What about Henriette?"
        # )

        # Example of a regular call

        llmfunction = LLMFunction(llm,
            "How many letters are there in this word : {word}",
            count=int,
        )
        llmfunction.invoke(word="blah")

        # Async call

        llmfunction.run_many([{"word": "strawberry"}, {"word": "banana"}])
    """
    parser: OutputParser

    def __init__(self, 
            llm: Runnable, 
            prompt: Union[str, List[Tuple], ChatPromptTemplate] = None, 
            max_retries: int = 2, 
            use_format_instructions: bool = True,
            pydantic_model: BaseModel = None, 
            **model_options
        ):
        """
        Initialize an LLMFunction instance to facilitate LLM function calls.

        Parameters:
            llm (Runnable): The underlying language model or runnable.
            prompt (str | List[Tuple] | ChatPromptTemplate, optional): The initial prompt which may be a system message,
                a list of (role, message) tuples, or a ChatPromptTemplate.
            max_retries (int, optional): Maximum number of retry attempts if output validation fails.
            use_format_instructions (bool, optional): If True, prepend parser-driven format instructions to the prompt.
            pydantic_model (BaseModel, optional): A prebuilt pydantic model for output parsing; if omitted,
                the model is built from model_options or inferred from the function signature.
            **model_options: Additional keyword arguments for configuring the pydantic model.
        
        Sets up an output parser and builds the prompt accordingly.
        """
        
        if max_retries is None: raise ValueError("max_retries must be specified")
        self.max_retries = max_retries

        self.pydantic_model = pydantic_model
        self.parser = OutputParser.from_mixed_input(pydantic_model=pydantic_model, **model_options)

        self.llm: Union[BaseChatModel, LLMWithTools] = llm  # NOTE : Can be changed to Runnable type if the current is too specialized, but this would requires some adjustments to be working with running tools in general
        self._build_prompt(prompt=prompt)

        if use_format_instructions:  # Add format instructions to the prompt
            self._add_format_instructions()

        
    def _add_format_instructions(self):
        format_instructions = self.parser.get_format_instructions_chat()
        self.prompt.insert(0, format_instructions)

    
    def _prompt_from_error(self, error: ValidationError) -> str:
        """
        Process the error and produces a prompt item to fix it.
        """
        return """
            Expected a structured output as specified in the system prompt. 
            To avoid this issue, your only output should be a structured message as specified in the system prompt
        """  # TODO : Try to extract the relevant pydantic error string
    

    def _build_prompt(self, prompt: Union[None, str, List[Tuple[str, Optional[str], Optional[str]]], ChatHistory, ChatPromptTemplate]):
        if not prompt:
            self.prompt = ChatHistory()
        elif isinstance(prompt, str):
            self.prompt: ChatHistory = ChatHistory().append("system", prompt, keep_variables=True)
        elif isinstance(prompt, list):  # List of tuples like [("system", "message"), ("ai", "response")]
            self.prompt: ChatHistory = ChatHistory().append(prompt, keep_variables=True)
        elif isinstance(prompt, ChatHistory):
            self.prompt: ChatHistory = prompt
        elif isinstance(prompt, ChatPromptTemplate):
            self.prompt: ChatHistory = ChatHistory().append(prompt.messages, keep_variables=True)  # Convert to ChatHistory
        else:
            raise ValueError(f"Unsupported prompt format. Must be a ChatPromptTemplate, str, or list of tuples. Instead got {prompt}.")


    def _prepare_local_prompt_and_input(self, input: Union[str, dict, PromptValue], kwargs: Optional[Dict[str, Any]]) -> Tuple[Dict[str, str], ChatHistory]:
        """
        Prepares the prompt and input for a new execution of the LLMFunction.
        """
        local_prompt = self.prompt

        if input is None: 
            input = dict()

        if isinstance(input, dict):
            if len(kwargs) > 0: input.update(kwargs)  # support adding inp values as kwargs

        elif isinstance(input, PromptValue):  # List[BaseMessage]
            local_prompt = local_prompt.copy()  # Only case where the copy method is needed.
            local_prompt.append(input.to_messages())
            input = dict()

        elif isinstance(input, str):
            if not local_prompt.exists("human"): local_prompt.create_type("human", "HumanMessage")
            local_prompt.append("human", input)
            input = dict()

        else:
            raise ValueError(f"Unsupported input format. Must be a dict, str, or list of BaseMessage. Instead got {input}.")
        
        return input, local_prompt


    def invoke_with_errors(
            self,
            input: Union[str, dict, PromptValue] = None, 
            max_retries: Optional[int] = None, 
            **kwargs
        ) -> LLMFunctionResult:
        """
        Run the invoke method, catches error and return a LLMFunctionResult object.
        """
        try:
            self.invoke(input=input, max_retries=max_retries, **kwargs)
        except Exception as e:
            return LLMFunctionResult(success=False, result=str(e))


    async def ainvoke(self, 
               input: Union[str, dict, PromptValue] = {}, 
               config: Optional[RunnableConfig] = None, 
               max_retries: Optional[int] = None, 
               sync_mode: bool = False,
               **kwargs
            ) -> BaseModel:
        """
            Runs the runnable, appends the errors, runs it again with the errors 
            until it is successful or max_retries is reached.

            /!\\ input values can be passed as the <input> dict or as <keyword arguments>.

            raise_errors: if set to True raise the exception if all attempts failed, return the result itself as a Type[BaseModel].
                          if set to False, always return a LLMFunctionResult with a success bool and the result or an error.
        """
        input, local_prompt = self._prepare_local_prompt_and_input(input=input, kwargs=kwargs)

        if max_retries is None: max_retries = self.max_retries
        current_retry_count = 0
        error_messages = list()

        while current_retry_count < max_retries:
            if current_retry_count > 0:
                logging.warning(f"Retrying : Attempt nb. {current_retry_count+1}")
                print(f"Retrying : Attempt nb. {current_retry_count+1}")

            try:
                if sync_mode:
                    return (local_prompt | self.llm | self.parser).invoke(input)
                else:
                    return await (local_prompt | self.llm | self.parser).ainvoke(input)

            except ToolInterrupt as interruption:
                raise interruption  # Interrupt and pass on the tool interrupt event, regardless of the raise_errors flag.

            except ValidationError as err:  # TODO : Use ValueError instead
                local_prompt.append("system", str(err))
                error_messages.append(str(err))

            current_retry_count += 1

        raise Exception(f"{error_messages}\n\nRetried too many times : {current_retry_count} times.")
    

    
    def invoke(self, 
        input: Union[str, dict, PromptValue] = {}, 
        config: Optional[RunnableConfig] = None, 
        max_retries: Optional[int] = None, 
        **kwargs
    ) -> BaseModel:
        if get_or_create_event_loop().is_running():
            return run_in_parallel_event_loop(future=self.ainvoke(input=input, config=config, max_retries=max_retries, sync_mode=True, **kwargs))
        else:
            return asyncio.run(main=self.ainvoke(input=input, config=config, max_retries=max_retries, sync_mode=True, **kwargs))


    def _handle_stream_event(
            self, 
            event: BaseMessage|StreamEvent, 
            buffer: str,
            previous_result: StableModel,
            disable_parsing: bool, 
            delta_mode: bool, 
        ) -> Tuple[str|StableModel|StreamEvent, StableModel, str]:
        """
        Regardless of parameters, events that are StreamEvents are simply streamed back as Tuple[1]

        If disable_parsing is True, return a str.
        Otherwise return a StableModel.

        disable_parsing = True => delta_mode ignored.
        """
        
        if isinstance(event, AIMessage):
            if disable_parsing:
                return event, previous_result, buffer  # Confirmation event = just stream the event
            else:
                # Ignore delta mode, buffer and all, just take the "result" message as it is and parse it into a result
                new_result = self.parser.parse_partial(event.content)
                return new_result, previous_result, buffer

        elif isinstance(event, AIMessageChunk):
            buffer += event.content
            
            if disable_parsing:
                return event.content, previous_result, buffer
            else:
                if delta_mode:
                    new_result = self.parser.parse_partial(buffer)
                    return previous_result.delta_from_newer(newer_model=new_result), new_result, buffer
                else:
                    new_result = self.parser.parse_partial(buffer)
                    return new_result, previous_result, buffer

        elif isinstance(event, ResetStream):
            # Reset the buffer
            return event, previous_result, ""
        
        else:
            return event, previous_result, buffer

    async def astream(self, 
               input: Union[str, dict, PromptValue] = {}, 
               config: Optional[RunnableConfig] = None, 
               max_retries: Optional[int] = None, 
               delta_mode: bool = False,
               disable_parsing: bool = False,
               sync_mode: bool = False,
               **kwargs
            ) -> AsyncGenerator[Union[StableModel, StreamEvent], StreamFlags]:
        """
            Streams the output as a StableModel.
            If max_retries is set, appends the errors, runs again with the errors integrated in the prompt
            until the function is successful or max_retries retries are reached.

            /!\\ input values can be passed as the <input> dict or as <keyword arguments>.

            max_retries: if set to None streams normally.
                         if set to an int, streaming might return a ResetStream flag that indicates that 
                         the content stream is beginning anew starting from the next item. External scripts typically want to reset their buffers.

            disable_parsing: if set to True, the LLMFunction will not parse the output into a StableModel. Instead, it will yield the raw output as a string.

            delta_model: if set to True, return the difference in content between the previous chunk and the current chunk as a StableModel. 
            The last streamed packed will contain all of the buffered data instead of the difference, so as to be used as a reference. 
            If set to False always stream all of the buffered data.
        """
        if not hasattr(self.llm, "astream"): raise Exception("The LLM provided to the constructor of the LLMFunction does not support asynchronous streaming.")
        input, local_prompt = self._prepare_local_prompt_and_input(input=input, kwargs=kwargs)
        if max_retries is None: max_retries = self.max_retries
        current_retry_count = 0
        error_messages = list()
        previous_result = self.parser.stable_pydantic_model()  # Used for delta mode

        while current_retry_count < max_retries:
            if current_retry_count > 0:
                logging.warning(f"Retrying : Attempt nb. {current_retry_count+1}")
                print(f"Retrying : Attempt nb. {current_retry_count+1}")

            try:
                buffer: str = ""

                if sync_mode:
                    for event in (local_prompt | self.llm).stream(input):
                        response, previous_result, buffer = self._handle_stream_event(
                            event=event, 
                            buffer=buffer, 
                            previous_result=previous_result,
                            disable_parsing=disable_parsing, 
                            delta_mode=delta_mode,
                        )
                        upstream = yield response
                else:
                    async for event in (local_prompt | self.llm).astream(input):
                        response, previous_result, buffer = self._handle_stream_event(
                            event=event, 
                            buffer=buffer, 
                            previous_result=previous_result,
                            disable_parsing=disable_parsing, 
                            delta_mode=delta_mode,
                        )
                        upstream = yield response

                if delta_mode:  # Yield the full buffer as the last message
                    yield self.parser.parse_partial(buffer)

                break  # End of streaming reached : Success

            except ToolInterrupt as interruption:
                raise interruption  # Interrupt and pass on the tool interrupt event, regardless of the raise_errors flag.

            except ValidationError as err:
                local_prompt.append("system", str(err))  # Add error to the local prompt so that the LLM won't make the same error
                error_messages.append(str(err))  # Keep track of the error so that they can be returned at the end
                logging.error("\n\nERROR\n\n", str(err), "\n\nBUFFER STATE\n\n", buffer, "\n\n")
                yield ResetStream(error=str(err))

            current_retry_count += 1

        Exception(f"{error_messages}\n\nRetried too many times : {current_retry_count} times.")


    def stream(self, 
               input: Union[str, dict, PromptValue] = {}, 
               config: Optional[RunnableConfig] = None, 
               max_retries: Optional[int] = None, 
               delta_mode: bool = False,
               disable_parsing: bool = False,
               **kwargs
            ) -> Generator[Union[StableModel, StreamEvent], StreamFlags, None]:
        """
            Streams the output as a StableModel.
            If max_retries is set, appends the errors, runs again with the errors integrated in the prompt
            until the function is successful or max_retries retries are reached.

            /!\\ input values can be passed as the <input> dict or as <keyword arguments>.

            max_retries: if set to None streams normally.
                         if set to an int, streaming might return a ResetStream flag that indicates that 
                         the content stream is beginning anew starting from the next item. External scripts typically want to reset their buffers.
        """
        async_gen_instance = self.astream(
            input=input, 
            config=config, 
            delta_mode=delta_mode, 
            disable_parsing=disable_parsing, 
            max_retries=max_retries, 
            sync_mode=True,  # Use the sync mode of the astream() method
            **kwargs
        )  # Initialize the async generator
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


    async def arun_many(self, 
                        inputs: List[Dict] = None, 
                        max_retries: int = None, 
                        raise_errors=True, 
                        use_threading=False, 
                        **kwargs
                    ) -> Union[List[BaseModel], BatchLLMFunctionResult]:
        """
        Run a batch call (many invoke() in parallel) using the current template, llm and model with a batch of input parameters.
        NOTE : This Method has the advantage of being compatible with Threading.

        /!\\ input values can be passed as the [insp] list or as [keyword arguments].
        Because we are processing a batch, <inputs> should be a List[Dict].
        Each keyword argument should be a " <key>=List[arg_i, ...] " where arg_i is the <key> parameter of the ith input of the batch.
        In other words, each keyword argument is the list of the arguments of that specific input keyword for each item of the batch.
        We should have the same value for each (len(arg) for arg in kwargs).
        """
        if max_retries is None: max_retries = self.max_retries

        if inputs:
            batch_size = len(inputs)
        else:  # no input
            batch_size = len(list(kwargs.values())[0])  # Using length of the first kwarg as a reference
        if inputs is None: inputs = [dict() for _ in range(batch_size)]

        # Process the kwargs
        if len(kwargs) != 0:
            for arg_name, batch_arg in kwargs.items():
                if len(batch_arg) != batch_size:
                    raise ValueError(f"All keyword arguments should have the same size as the batch. Got {len(batch_arg)} arguments for '{arg_name}' while the ref batch size is {batch_size}.")
                # Compose the inp dicts with each argument from the batch
                for input, arg in zip(inputs, batch_arg):
                    input.update({arg_name: arg})

        results: List[LLMFunctionResult] = [None] * len(inputs)

        if use_threading:
            # Non-blocking: Use asyncio.to_thread to run tasks in a thread-pool without blocking the event loop
            async def run_in_thread_async(input):
                return await asyncio.to_thread(
                    self.run, input, max_retries=max_retries, raise_errors=False
                )
            # Launch tasks asynchronously using asyncio.gather
            tasks = [run_in_thread_async(input) for input in inputs]
            results = await asyncio.gather(*tasks)
        else:
            # Just run the batch via asyncio
            results = await asyncio.gather(*(self.ainvoke(input, max_retries=max_retries, raise_errors=False) for input in inputs))

        if raise_errors:

            errors = "\n"

            if any(lr is None or not lr.success for lr in results):
                errors += "\n".join(f"Error in task {i}: {"Other error" if lr is None else lr.result}" if lr is None or not lr.success else f"Task {i} completed successfully" for i, lr in enumerate(results))
                raise Exception(errors)
            
            return [res.result for res in results]  # List[BaseModel]
        
        else:  # raise_errors = False
            success = all(lr.success for lr in results)
            return BatchLLMFunctionResult(success, results)
        

    def run_many(self, 
                 inputs: List[Dict]=None, 
                 max_retries: int=None, 
                 raise_errors=True, 
                 use_threading=False, 
                 **kwargs
            ) -> Union[List[BaseModel], BatchLLMFunctionResult]:
        """
        Synchronous wrapper around arun_many().
        NOTE : This Method has the advantage of being compatible with Threading.
        """
        try:
            if get_or_create_event_loop().is_running():  # Running inside an async loop
                future = asyncio.ensure_future(self.arun_many(inputs=inputs, max_retries=max_retries, raise_errors=raise_errors, use_threading=use_threading, **kwargs))
                return asyncio.get_event_loop().run_until_complete(future)
                
        finally:
            return asyncio.run(self.arun_many(inputs=inputs, max_retries=max_retries, raise_errors=raise_errors, use_threading=use_threading, **kwargs))
        