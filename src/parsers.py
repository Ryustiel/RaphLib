"""
The output parsers get your format instructions, 
tell the llm what kind of output to produce, 
and extract the information you want in an output pydantic.
"""
import logging

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
from abc import ABC, abstractmethod

from langchain.output_parsers import PydanticOutputParser as LangchainPydanticOutputParser
from langchain_core.messages import BaseMessage
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig

from .helpers import escape_characters, repair_list
from .stables import StableModel, pydantic_model_from_options

DEFAULT_RAISE_MIXED_INPUT_ERROR = True

class OutputParser(Runnable, ABC):
    """
    A general class for output parsers used in LLMFunctions.
    They provide specific chunks of prompts (can include examples) and parse the output of the LLM based on which prompt has been used.
    """
    def __init__(self, pydantic_model: BaseModel):
        self.pydantic_model: BaseModel = pydantic_model
        self.stable_pydantic_model: StableModel = StableModel.from_model(self.pydantic_model)

    @abstractmethod
    def parse_partial(self, partial_json: str) -> StableModel:
        """
        Parse the partial input into an instance of a Type[StableModel] corresponding to the parser's original parameters.
        """
        pass

    @abstractmethod
    def parse(self, json_input: str) -> BaseModel:
        """
        Parse the json input into an instance of a Type[BaseModel] corresponding to the parser's original parameters.
        """
        pass

    @abstractmethod
    def get_format_instructions_chat(self) -> List[BaseMessage]:
        pass

    def invoke(self, input: Union[str, BaseMessage] = {}, config: Optional[RunnableConfig] = None) -> BaseModel:
        """
        Parameters:
        - input: The output from the LLM.

        Returns:
        - BaseModel: Parsed output conforming to the expected_type.
        """
        if isinstance(input, BaseMessage):
            input = input.content
        # TODO Manage logprobs here from the llm output : pass them over to the output step as BaseModel, logprob (should be something about passing on the config parameter somehow?)
        # OR ! Simply manage the logprobs in the LLMFunction call itself so that events can be emitted for that.
        return self.parse(input)

    @classmethod
    def from_example(cls, **model_options) -> 'OutputParser':
        """
        Create a FastOutputParser instance from the model options (example) if there is only one field.
        Otherwise, create a DetailedOutputParser out of the options.
        """
        pydantic_model = pydantic_model_from_options(**model_options)
        # fast single value output
        if len(model_options) == 1 and next(iter(model_options.values())) is not str and next(iter(model_options.values())) != str:  # str values should be parsed as a json output.
            return FastOutputParser(pydantic_model=pydantic_model)
        else:  # json composed output  
            return DetailedOutputParser(pydantic_model=pydantic_model)
    
    @classmethod
    def from_pydantic_model(cls, pydantic_model: BaseModel) -> 'OutputParser':
        """
        Create a DetailedOutputParser instance from a pydantic model.
        """
        return DetailedOutputParser(pydantic_model)
    
    @classmethod
    def from_mixed_input(cls, pydantic_model: Optional[BaseModel] = None, raise_mixed_input_error = DEFAULT_RAISE_MIXED_INPUT_ERROR, **model_options) -> 'OutputParser':
        """
        Create a FastOutputParser instance from a pydantic model.
        If a pydantic model is provided, it will be used. Otherwise, it will create a model from the given options.

        Raises errors if both options and a pydantic model are provided.
        """
        if pydantic_model:
            if len(model_options) >= 1:
                if raise_mixed_input_error:
                    raise ValueError(f"Provided {len(model_options)} model options and a pydantic model in the same time. Both options are mutually exclusive, it's either a prebuilt model or model options.")
                else:
                    return cls.from_pydantic_model(pydantic_model=pydantic_model)
            # prebuilt pydantic model
            else:
                return cls.from_pydantic_model(pydantic_model=pydantic_model)
        else:
            return cls.from_example(**model_options)


class FastOutputParser(OutputParser):
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
    def __init__(self, pydantic_model: BaseModel):
        super().__init__(pydantic_model=pydantic_model)
        self.restriction = None  # Example for the literal case
        self.expected_type: Tuple[type, type] = self._get_expected_type()

    def _get_field_name(self):
        """Get the name of the only field in the pydantic model.
        This is used to initialize the model with that field once the input has been validated.
        """
        return next(iter(self.pydantic_model.model_fields.keys()))
    
    def _get_expected_type(self) -> Tuple[type, type]:
        """
        Return a tuple (is_literal, type)
        If the input is a literal, the type of the input should be in the list.
        """
        first_field_name = self._get_field_name()
        field_info = self.pydantic_model.model_fields[first_field_name]
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
    
    def parse_partial(self, partial_json: str) -> StableModel:
        """
        Extract the content as a json and parse it into a StableModel instead.
        """
        formatted_json = self._extract_json(text_input=partial_json)
        try:
            return self.stable_pydantic_model.model_validate_json(formatted_json)
        except ValidationError as e:
            return self.stable_pydantic_model()  # Return an empty version of the pydantic model as a default

    def parse(self, json_input: str) -> BaseModel:
        """
        Extract the content as a json and parse it into a BaseModel. This has the same effect as parse_partial, but the return type is not the same.
        NOTE : The parameter name "json_input" can be misleading since a FastOutputParser actually expects a NON json formatted string.
        """
        formatted_json = self._extract_json(text_input=json_input)
        return self.pydantic_model.model_validate_json(formatted_json)  # json_input is typically not a formatted json

    def _extract_json(self, text_input: str) -> str:
        """
        Attempts at parsing the input into a fixed type.

        Parameters:
        - json_input: A valid json string

        Returns:
        - BaseModel: Parsed output conforming to the expected_type.
        """
        try:
            if text_input == "":
                return "{}"
            if self.expected_type[0] is None or self.expected_type[0] is Literal:  # Handling string literals
                
                if self.expected_type[1] is str:
                    text_input = f"\"{text_input}\""  # turns blah into "blah" for json serialization.
                
                elif self.expected_type[1] is bool:
                    match text_input[0].upper():
                        case "F":
                            text_input = "False"
                        case "T":
                            text_input = "True"
                        case _:
                            text_input = None

                elif self.expected_type[1] is float:  # Removing trailing dots
                    if text_input[-1] in (".", ","):
                        text_input = text_input[:-1] 

            elif (
                self.expected_type[0] is list 
                and (self.expected_type[1] is str or self.expected_type[1] is bool)
                and text_input[0] == "[" and text_input[1] == "'"
            ):
                text_input = repair_list(text_input)
            
            if text_input:
                return "{\"" + f"{self._get_field_name()}\": {text_input}" + "}"
            else:
                return "{}"
        except Exception as e:
            raise Exception(f"{e.__class__}: {e}.\n Error while parsing LLM output with a FastParser. Probably caused by a wrong llm input like a float instead of an int. Check FastOutputParser._extract_json.")
        
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
        

class DetailedOutputParser(OutputParser):  # A wrapper on the LangchainPydanticOutputParser for managing format instructions and getting logprobs from calls.
    """
    Unlike the FastOutputParser, this basically calls the pydantic json validation method.
    This works when a LLM is asked to produce a structured json output, whose format was previously agreed upon when instanciating LLMFunction.
    """
    def __init__(self, pydantic_model: BaseModel):
        super().__init__(pydantic_model=pydantic_model)
        self.langchain_pydantic_output_parser = LangchainPydanticOutputParser(pydantic_object=pydantic_model)

    def parse(self, json_input: str) -> BaseModel:
        """
        Parse the partial JSON representation using Pydantic's built in validation method.
        """
        return self.langchain_pydantic_output_parser.invoke(json_input)
    
    def parse_partial(self, partial_json: str) -> StableModel:
        """
        Parse the partial JSON representation using the StableModel's built in parser.
        """
        return self.stable_pydantic_model.from_partial_json(partial_json)
    
    def get_format_instructions_chat(self) -> List[BaseMessage]:
        """
        Create a format instruction using Langchain's Pydantic Output Parser.
        """
        return [("system", self.langchain_pydantic_output_parser.get_format_instructions())]
