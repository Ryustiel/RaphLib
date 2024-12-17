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
    get_args,
    get_origin,
)
from pydantic import BaseModel, Field, create_model
from pydantic_core import ValidationError

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import BaseMessage
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.prompt_values import PromptValue
from langchain_core.language_models.chat_models import BaseChatModel

from .prompts import ChatHistory
from .tools import ToolInterrupt, LLMWithTools
from .helpers import escape_characters
from .stables import StableModel, pydantic_model_from_options


LLMStreamFlags = Literal["cancel", "interrupt"]


class StreamReset(BaseModel):
    """
    Flag raised when a stream has been reset.
    """
    error: str


class FastOutputParser(Runnable):
    """
    Parse the output of a LLM who does not provide a json output, 
    and instead provide the desired output directly as a string value. (like "true", "12.5", or "[1, 2, 4]", etc...)
    Attempts to directly parse the output using pythonic ways instead of relying on pydantic json validation.

    Example :

    fnc = LLMFunction(LLM,
        "Create a mask that represents for each word if it is a verb : {message}",
        words=["word", ...],
        mask=[True, ...]
    )

    fnc.prompt.pretty_print({"message": "Test"})

    messages = ["I want to eat cakes", "How many cakes are in that truck ?", "Shit won't hold that cow", "Two trucks are necessary to keep on working."]
    responses = fnc.run_many(message=messages)
    print("\n".join([f"{message}\t\t\t>> {classe.words} {classe.mask}" for (message, classe) in zip(messages, responses)]))
    """
    def __init__(self, pydantic_object: BaseModel):
        self.pydantic_object: BaseModel = pydantic_object
        self.restriction = None  # Example for the literal case
        self.expected_type: Tuple[type, type] = self._get_expected_type()

    def _get_field_name(self):
        """Get the name of the only field in the pydantic model.
        This is used to initialize the model with that field once the input has been validated.
        """
        return next(iter(self.pydantic_object.model_fields.keys()))
    
    def _get_expected_type(self) -> Tuple[type, type]:
        """
        Return a tuple (is_literal, type)
        If the input is a literal, the type of the input should be in the list.
        """
        first_field_name = self._get_field_name()
        field_info = self.pydantic_object.model_fields[first_field_name]
        field_type = field_info.annotation
        if get_origin(field_type) is Literal:
            type_args = get_args(field_type)
            self.restriction = type_args
            if type_args:
                return Literal, type(type_args[0])  # Return type of the first item of Literal
            else:
                return Literal, None  # Return None if the list is empty
        elif get_origin(field_type) is list:
            type_args = get_args(field_type)
            if type_args:
                return list, type_args[0]
            else:
                return list, None
              
        # If not a Literal, return the type directly
        return None, field_type

    def invoke(self, input: Union[str, BaseMessage] = {}, config: Optional[RunnableConfig] = None) -> BaseModel:
        """
        Parameters:
        - input: The output from the LLM.

        Returns:
        - BaseModel: Parsed output conforming to the expected_type.
        """
        if isinstance(input, BaseMessage):
            input = input.content
        try:
            if input == "":
                raise ValueError("No input was provided.")
            if(
                (self.expected_type[0] is None or self.expected_type[0] is Literal) 
                and (self.expected_type[1] is str or self.expected_type[1] is bool)
            ):  # Is it a properly formatted string
                if input[0] == "\"":
                    input.replace("'", "\"")  # turns 'blah' into "blah" for json serialization.
                elif input[-1] != "\"":
                    input = f"\"{input}\""  # turns blah into "blah" for json serialization.
                # else: input already has " "
            elif (
                self.expected_type[0] is list 
                and (self.expected_type[1] is str or self.expected_type[1] is bool)
                and input[0] == "[" and input[1] == "'"
            ):
                print(input)
                input.replace("'", "\"")  # Change ['a', 'b'] to ["a", "b"] for json serialization.
            json_string = "{\"" + f"{self._get_field_name()}\": {input}" + "}"

            return self.pydantic_object.model_validate_json(json_string)
        except ValidationError as err:
            raise ValueError(f"Parsed output didn't match the expected schema: {err}")
        
    def get_format_instructions_chat(self) -> List[BaseMessage]:
        first_part = "You are an analyst who only responds with"
        if self.expected_type[0] is list:
            if self.expected_type[1] is str: return [("system", f"{first_part} a list of strings."), ("ai", "[\"abc\", \"def\"]")] 
            elif self.expected_type[1] is int: return [("system", f"{first_part} a list of integers."), ("ai", "[1, 2, 3]")]
            elif self.expected_type[1] is float: return [("system", f"{first_part} a list of floats."), ("ai", "[1.0, 2.0, 3.0]")]
            elif self.expected_type[1] is bool: return [("system", f"{first_part} a list of either of \"True\" or \"False\"."), ("ai", "[\"True\", \"False\", \"True\"]")]
        elif self.expected_type[0] is None:
            if self.expected_type[1] is int: return [("system", f"{first_part} a single integer."), ("ai", "0")]
            elif self.expected_type[1] is float: return [("system", f"{first_part} a single float."), ("ai", "1.0")]
            elif self.expected_type[1] is bool: return [("system", f"{first_part} True or False."), ("ai", "False")]
        elif self.expected_type[0] is Literal:
            if self.expected_type[1] is int: return [("system", f"{first_part} one of the following integers : {self.restriction}"), ("ai", str(self.restriction[0]))]
            elif self.expected_type[1] is float: return [("system", f"{first_part} one of the following floats : {self.restriction}"), ("ai", str(self.restriction[0]))]
            elif self.expected_type[1] is str: return [("system", f"{first_part} one of the following values : {self.restriction}"), ("ai", self.restriction[0])]
        raise ValueError(f"Did not implement get_format_instructions_chat for the type \"{self.expected_type}\".")
        

class DetailedOutputParser(PydanticOutputParser):  # A wrapper on the PydanticOutputParser for managing format instructions and getting logprobs from calls.
    """
    Unlike the FastOutputParser, this basically calls the pydantic json validation method.
    This works when a LLM is asked to produce a structured json output, whose format was previously agreed upon when instanciating LLMFunction.
    """
    def invoke(self, input: Union[str, BaseMessage], config: Optional[RunnableConfig] = None) -> Any:
        # input is the output of the llm call
        # TODO Manage logprobs here from the llm output : pass them over to the output step as BaseModel, logprob
        return super().invoke(input)
    
    def get_format_instructions_chat(self) -> List[BaseMessage]:
        return [("system", super().get_format_instructions())]


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
    An interface to create LLM function calls with minimal effort.

    Prompt can be a string (interpreted as a system prompt), a [("type", "message"),] message list, or a ChatPromptTemplate.
    Kwargs are parameters for the pydantic model, except the special "pydantic_model" kwarg that accepts a prebuilt pydantic model.

    raise_errors: if True raise the exception if there is any error while invoking, return only the result as a single value (BaseModel).
                  if False return a tuple with a success bool and the result or an error (LLMFunctionResult).
    
    Functionalities : 
    - Ensures the output of the LLM is conform to a pydantic model.
    - Provide the output as a pydantic instance of that model.
    - Retry when the reply is wrong.
    - Provide extra system prompt messages to help prevent the model from failing multiple times. TODO
    - Provide statistics about the replies from the model if logprobs are available. TODO
    - Provide syntactic tools to create LLM functions and pydantic models easily.
    - Simplify the execution of batch calls (parallelized) -- from a batch of parameters.

    Examples :

    # Function call using an external ChatPromptTemplate

    llmfunction = LLMFunction(llm,
        "Note all the names mentioned in this conversation, as well as the number of questions that were asked.",
        names=["raph", ...],
        number_of_questions=5,
    )

    conversation = ChatHistory([
        ("sam", "bye felicia."),
        ("sam", "{sentence}")
    ])

    (conversation | llmfunction).invoke({"sentence": "What about Henriette ?"})

    # The llm will be called on (
    #   "Note all the names mentioned in this conversation, as well as the number of questions that were asked."
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
    def __init__(self, 
                 llm: Runnable, 
                 prompt: Union[str, List[Tuple], ChatPromptTemplate] = None, 
                 max_retries: int = 2, 
                 raise_errors: bool = True, 
                 use_format_instructions: bool = True,
                 pydantic_model: BaseModel = None, 
                 **model_options):
        
        if max_retries is None: raise ValueError("max_retries must be specified")

        self.raise_errors = raise_errors
        self.max_retries = max_retries

        # Build streaming flags
        self.is_streamable: bool = hasattr(llm, "stream")

        if pydantic_model:
            if len(model_options) >= 1:
                raise ValueError(f"Provided {len(model_options)} model options and a pydantic model in the same time. Both options are mutually exclusive, it's either a prebuilt model or model options.")
            # prebuilt pydantic model
            else:
                self.model = pydantic_model
                self.parser = DetailedOutputParser(pydantic_object=self.model)
        else:
            self.model = pydantic_model_from_options(**model_options)
            # fast single value output
            if len(model_options) == 1 and next(iter(model_options.values())) is not str and next(iter(model_options.values())) != str:  # str values should be parsed as a json output.
                self.parser = FastOutputParser(pydantic_object=self.model)
            else:  # json composed output  
                self.parser = DetailedOutputParser(pydantic_object=self.model)

        self.llm: Union[BaseChatModel, LLMWithTools] = llm  # NOTE : Can be changed to Runnable type if too specialized, but this would requires some adjustments to be work with running tools

        if not prompt:
            self.prompt = ChatHistory()
        elif isinstance(prompt, str):
            self.prompt: ChatHistory = ChatHistory().append("system", prompt, keep_variables=True)
        elif isinstance(prompt, list):  # List of tuples like [("system", "message"), ("ai", "response")]
            self.prompt: ChatHistory = ChatHistory().append(prompt)
        elif isinstance(prompt, ChatHistory):
            self.prompt: ChatHistory = prompt
        elif isinstance(prompt, ChatPromptTemplate):
            self.prompt: ChatHistory = ChatHistory().append(prompt.messages)  # Convert to ChatHistory
        else:
            raise ValueError(f"Unsupported prompt format. Must be a ChatPromptTemplate, str, or list of tuples. Instead got {prompt}.")

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


    def _prepare_local_prompt_and_input(self, input: Union[str, dict, PromptValue], kwargs: Optional[Dict[str, Any]]) -> Tuple[Dict[str, str], ChatHistory]:
        """
        Prepares the prompt and input for a new execution of the LLMFunction.
        """
        local_prompt = self.prompt

        if not input: 
            input = dict()

        elif isinstance(input, PromptValue):  # List[BaseMessage]
            local_prompt = local_prompt.copy()  # Only case where the copy method is needed.
            local_prompt.append(input.to_messages())
            input = dict()

        elif isinstance(input, str):
            if not local_prompt.exists("human"): local_prompt.create_type("human", "HumanMessage")
            local_prompt.append("human", input.content)
            input = dict()

        elif isinstance(input, dict):
            if len(kwargs) > 0: input.update(kwargs)  # support adding inp values as kwargs

        else:
            raise ValueError(f"Unsupported input format. Must be a dict, str, or list of BaseMessage. Instead got {input}.")
        
        return input, local_prompt


    def invoke(self, 
               input: Union[str, dict, PromptValue] = None, 
               config: Optional[RunnableConfig] = None, 
               max_retries: Optional[int] = None, 
               raise_errors: bool = None, 
               **kwargs
            ) -> Union[BaseModel, LLMFunctionResult]:
        """
            Runs the runnable, appends the errors, runs it again with the errors 
            until it is successful or max_retries is reached.

            /!\\ input values can be passed as the <input> dict or as <keyword arguments>.

            raise_errors: if set to True raise the exception if all attempts failed, return the result itself as a Type[BaseModel].
                          if set to False, always return a LLMFunctionResult with a success bool and the result or an error.
        """
        # PARSING ARGUMENTS
        input, local_prompt = self._prepare_local_prompt_and_input(input=input, kwargs=kwargs)

        if raise_errors is None: raise_errors = self.raise_errors
        if max_retries is None: max_retries = self.max_retries

        # RUNNING CHAIN WITH RETRY LOOP
        current_retry_count = 0
        error_messages = list()

        result: BaseModel = None
        while current_retry_count < max_retries:
            if current_retry_count > 0:
                logging.warning(f"Retrying : Attempt nb. {current_retry_count+1}")

            try:
                result = (local_prompt | self.llm | self.parser).invoke(input)
                break  # If no error, break the loop.

            except ValidationError as err:  # TODO : Use ValueError instead
                error = str(err)
                local_prompt.append("system", error)
                error_messages.append(error)

            except ToolInterrupt as interruption:
                raise interruption  # Interrupt and pass on the tool interrupt event, regardless of the raise_errors flag.
            
            except Exception as err:
                if raise_errors: raise err
                else: return LLMFunctionResult(False, err)

            if current_retry_count >= max_retries:
                err = Exception(f"{error_messages}\n\nRetried too many times : {current_retry_count} times.")
                if raise_errors: 
                    raise err
                else: 
                    return LLMFunctionResult(False, err)

            current_retry_count += 1

        # the flag "raise_errors" also affects the return type of a valid result 
        if raise_errors: 
            return result
        else: 
            return LLMFunctionResult(True, result)
        

    async def astream_raw(self, 
               input: Union[str, dict, PromptValue] = None, 
               config: Optional[RunnableConfig] = None, 
               max_retries: int = 1, 
               raise_errors: bool = None, 
               **kwargs
            ) -> AsyncGenerator[Union[str, StreamReset], LLMStreamFlags]:
        """
            Streams the output as a str textdelta or as a StableModel (depending on the configuration of the LLMFunction).
            If max_retries is set, appends the errors, runs again with the errors integrated in the prompt
            until the function is successful or max_retries retries are reached.

            /!\\ input values can be passed as the <input> dict or as <keyword arguments>.

            max_retries: if set to None streams normally.
                         if set to an int, streaming might return a StreamReset flag that indicates that 
                         the content stream is beginning anew starting from the next item. External scripts typically want to reset their buffers.

            raise_errors: if set to True raise the exception if all attempts failed, yield stream items directly a Type[StableModel] or a (textdelta) str.
                          if set to False, always return a LLMFunctionResult with a success bool and the result or an error.
        """
        if not self.is_streamable: raise Exception("The LLM provided to the constructor of the LLMFunction does not support streaming.")

        input, local_prompt = self._prepare_local_prompt_and_input(input=input, kwargs=kwargs)

        if raise_errors is None: raise_errors = self.raise_errors
        if max_retries is None: max_retries = self.max_retries

        # RUNNING CHAIN WITH RETRY LOOP
        current_retry_count = 0
        error_messages = list()

        while current_retry_count < max_retries:
            if current_retry_count > 0:
                logging.warning(f"Retrying : Attempt nb. {current_retry_count+1}")

            try:
                for chunk in (local_prompt | self.llm).stream(input):
                    # chunk: BaseMessage
                    upstream = yield chunk.content

                    # TODO : Act upon upstream

                break  # End of streaming reached : Success

            except ToolInterrupt as interruption:
                raise interruption  # Interrupt and pass on the tool interrupt event, regardless of the raise_errors flag.
            
            except Exception as err:
                if raise_errors: 
                    raise err

                else: 
                    upstream = yield StreamReset(error=str(err))

                    # TODO : Act upon upstream

                current_retry_count += 1

        if current_retry_count >= max_retries:
            err = Exception(f"{error_messages}\n\nRetried too many times : {current_retry_count} times.")
            if raise_errors: 
                raise err
            else: 
                yield StreamReset(error=str(err))

        # else : Streaming was successful

    async def astream(self, 
               input: Union[str, dict, PromptValue] = None, 
               config: Optional[RunnableConfig] = None, 
               max_retries: int = 1, 
               raise_errors: bool = None, 
               **kwargs
            ) -> AsyncGenerator[Union[StableModel, StreamReset], LLMStreamFlags]:
        """
            Streams the output as a StableModel.
            If max_retries is set, appends the errors, runs again with the errors integrated in the prompt
            until the function is successful or max_retries retries are reached.

            /!\\ input values can be passed as the <input> dict or as <keyword arguments>.

            max_retries: if set to None streams normally.
                         if set to an int, streaming might return a StreamReset flag that indicates that 
                         the content stream is beginning anew starting from the next item. External scripts typically want to reset their buffers.

            raise_errors: if set to True raise the exception if all attempts failed, yield stream items directly a Type[StableModel] or a (textdelta) str.
                          if set to False, always return a LLMFunctionResult with a success bool and the result or an error.
        """
        generator = self.astream_raw()
        async for chunk in generator:
            if isinstance(chunk, StreamReset):
                upstream = yield chunk  # Passing on the StreamReset flag
            else:
                upstream = yield self.stable_model(chunk)
            await generator.asend(upstream)

    def stream_raw(self, 
               input: Union[str, dict, PromptValue] = None, 
               config: Optional[RunnableConfig] = None, 
               max_retries: int = 1, 
               raise_errors: bool = None, 
               **kwargs
            ) -> Generator[Union[StableModel, StreamReset], LLMStreamFlags, None]:
        """
            Streams the output as a StableModel.
            If max_retries is set, appends the errors, runs again with the errors integrated in the prompt
            until the function is successful or max_retries retries are reached.

            /!\\ input values can be passed as the <input> dict or as <keyword arguments>.

            max_retries: if set to None streams normally.
                         if set to an int, streaming might return a StreamReset flag that indicates that 
                         the content stream is beginning anew starting from the next item. External scripts typically want to reset their buffers.

            raise_errors: if set to True raise the exception if all attempts failed, yield stream items directly a Type[StableModel] or a (textdelta) str.
                          if set to False, always return a LLMFunctionResult with a success bool and the result or an error.
        """
        # Wrapper around stream
        yield ""

    def stream(self, 
               input: Union[str, dict, PromptValue] = None, 
               config: Optional[RunnableConfig] = None, 
               max_retries: int = 1, 
               raise_errors: bool = None, 
               **kwargs
            ) -> Generator[Union[StableModel, StreamReset], LLMStreamFlags, None]:
        """
            Streams the output as a StableModel.
            If max_retries is set, appends the errors, runs again with the errors integrated in the prompt
            until the function is successful or max_retries retries are reached.

            /!\\ input values can be passed as the <input> dict or as <keyword arguments>.

            max_retries: if set to None streams normally.
                         if set to an int, streaming might return a StreamReset flag that indicates that 
                         the content stream is beginning anew starting from the next item. External scripts typically want to reset their buffers.

            raise_errors: if set to True raise the exception if all attempts failed, yield stream items directly a Type[StableModel] or a (textdelta) str.
                          if set to False, always return a LLMFunctionResult with a success bool and the result or an error.
        """
        yield ""

    async def arun_many(self, 
                        inputs: List[Dict] = None, 
                        max_retries: int = None, 
                        raise_errors=None, 
                        use_threading=False, 
                        **kwargs
                    ) -> Union[List[BaseModel], BatchLLMFunctionResult]:
        """
        Run a batch call (many invoke() in parallel) using the current template, llm and model with a batch of input parameters.

        /!\\ input values can be passed as the [insp] list or as [keyword arguments].
        Because we are processing a batch, <inputs> should be a List[Dict].
        Each keyword argument should be a " <key>=List[arg_i, ...] " where arg_i is the <key> parameter of the ith input of the batch.
        In other words, each keyword argument is the list of the arguments of that specific input keyword for each item of the batch.
        We should have the same value for each (len(arg) for arg in kwargs).
        """
        if raise_errors is None: raise_errors = self.raise_errors
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
                 raise_errors=None, 
                 use_threading=False, 
                 **kwargs
            ) -> Union[List[BaseModel], BatchLLMFunctionResult]:
        """ Synchronous wrapper around arun_many()."""
        try:
            if asyncio.get_event_loop().is_running():  # Running inside an async loop
                future = asyncio.ensure_future(self.arun_many(inputs=inputs, max_retries=max_retries, raise_errors=raise_errors, use_threading=use_threading, **kwargs))
                return asyncio.get_event_loop().run_until_complete(future)
                
        finally:
            return asyncio.run(self.arun_many(inputs=inputs, max_retries=max_retries, raise_errors=raise_errors, use_threading=use_threading, **kwargs))
        
"""
Streaming that makes sense : 
- When streaming a tool call mixed with a chat, streams the text as an event, 
then the pydantic objects as events, and potential "reset" events in case of the model fails and tries again with a structured output.
- Tools can be made compatible with streaming if they're defined with a special "streamable" decorator. 
They should then act like a "streamable pydantic" async generator.
- The weight of the output parsing should be on the streamed object model. Use a base "streamable pydantic" object.
- When streaming a structured response, sends the structured pydantic object repetitively, with missing fields.

1. Create and test a homeostatic pydantic object
2. Check if a target pydantic model (that is generated through our functions) can be "made partial" automatically.
3. Check the streaming methods that come with langchain and implement them with the new pydantic streaming method (with the detailed output parser for now).
4. Make it compatible with the fast output parser.
5. Analyze the tool decorator and look for ways to make it compatible with streaming the tool output (if it's a structured object, otherwise just wait).
6. Move the "stream messages" gimmick into its own separate method, so that the change in behavior and return type is clearer.
7. Either way, the model will have to wait for the full tool output. At this point, let's make the LLMTool streaming event based. (Create a method for just that)

A LLMFunction could take in a "LLMTool compatible LLM object", which has "multi output" enabled, and so return the multi output, with the parsed response as a... string ?
Or maybe as a compound special output like "Chat Catchup" + "Structured Output"

8. Implement the Chat Catchup object.
"""