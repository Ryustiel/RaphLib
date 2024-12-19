"""
Stable models are pydantic models that are designed to parse themselves into a "closest" relative on model_validate, 
nulling all missing fields.
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
    Type, 
    Optional, 
    get_args,
    get_origin,
)
from pydantic import BaseModel, Field, create_model
from pydantic_core import ValidationError

from .helpers import get_all_fields_as_optional, diff_dict, repair_json

# ================================================================= STABLE MODEL =================================

class StableModel(BaseModel):
    """
    A StableModel is a pydantic model that has a defined structure and can be parsed into a "closest" relative,
    usually with None values for the missing fields.
    Stable models can handle partial json objects.
    """

    def delta_from_newer(self, newer_model: 'StableModel', compute_diff_string: bool = False) -> 'StableModel':
        """
        Will produce a StableModel that only has the difference between the current content and the new content.
        All the other fields will be set to None.

        From 2 pydantic models (with all fields optional) and create a new instance of the model class that only contains the differences. 
        
        Example :
        {"a":{"b": "1", "c":["1"]}} diff {"a":{"b": "1", "c":["1", "2"], "d":"2"}} => {"a":{"c": ["2"], "d": "2"}}

        A difference is always positive, input 2 always has at least as many items that input 1 has (it's from a similar json with only additional keys).
        """
        current_attributes: dict = self.model_dump(exclude_none=True)
        newer_attributes: dict = newer_model.model_dump(exclude_none=True)

        delta_attributes = diff_dict(exclude_dict=current_attributes, data_dict=newer_attributes, compute_diff_string=compute_diff_string)
        delta_model = self.model_validate(delta_attributes)  # Generate a new model based on the difference in attributes

        return delta_model

    @classmethod
    def from_partial_json(cls, trucated_json: str) -> 'StableModel':
        """
        Complete a trucated JSON string and return an instance of StableModel.
        """
        repaired_json = repair_json(truncated_json=trucated_json)

        try:
            return cls.model_validate_json(repaired_json)
        except ValidationError as e:
            raise e
       
    @classmethod
    def from_model(cls, model: Type[BaseModel]) -> Type['StableModel']:
        """
        Build a stable model class from a reference pydantic model.
        """
        optionalized_fields = get_all_fields_as_optional(model)
        new_model: BaseModel = create_model(model.__name__, __base__=StableModel, **optionalized_fields)
        return new_model

# ================================================================= CREATE MODEL =================================

# NOTE : These types are supported explicitly in the extract_fields methods, but also in the functions.OutputParser methods.
# If you expand one of these lists you have to also make changes to the extract_fields fucntion below AND the .functions.FastOutputParser's methods
ALLOWED_TYPES = (str, int, float, bool)
ALLOWED_TYPE_KEYWORDS = (str, int, float, bool, list, List, List[str], List[int], List[float], List[bool])

def _check_list_type(key: str, li: list, target_type: type) -> bool:
    # Check if the list is conform to create a pydantic model
    if not target_type in ALLOWED_TYPES:
        raise ValueError(f"All choices should be of one of the allowed types [{ALLOWED_TYPES}] at @{key} : {li} ({target_type})")
    if not all(isinstance(i, target_type) for i in li):
        raise ValueError(f"All of the choices should be of the same type at @{key} : {li}")
    return True

def extract_fields(**example) -> Dict[str, Any]:
    """
    INFO : You can't mix types in literals defined through this function.
    All models with literals should be created through this function so they have this property as well.
    """
    # Create a pydantic model from the example
    fields: Dict[str, Any] = {}
    for key, value in example.items():

        if isinstance(value, list):
            if len(value) > 1:  # The input was not [] or ["item"]
                # Free typed list
                if any(isinstance(i, type(...)) for i in value):  # The input was something like [example1, example2, ...]
                    # Determine the type of the values
                    values_no_ellipsis = [i for i in value if not isinstance(i, type(...))]  # Filter out ellipsis
                    if len(values_no_ellipsis) >= 1:  # Ensure the input was not [...]
                        target_type = type(values_no_ellipsis[0])
                        if _check_list_type(key, values_no_ellipsis, target_type):  # The whole list is target_type
                            fields[key] = (List[target_type], Field(..., example=values_no_ellipsis))
                            continue

                # Choices
                else:  # The input was something like [choice1, choice2]
                    target_type = type(value[0])
                    if _check_list_type(key, value, target_type):  # The whole list is target_type : Prevents types from being mixed
                        fields[key] = (Literal[tuple(value)], Field(...,))
                        continue
                
        elif isinstance(value, tuple) and all(isinstance(i, str) for i in value):
            # string with multiple examples
            fields[key] = (str, Field(..., example=', '.join(value)))
            continue
        elif type(value) in ALLOWED_TYPES:  # Case example of a type
            fields[key] = (type(value), Field(..., example=value))
            continue
        elif value in ALLOWED_TYPE_KEYWORDS:  # Case the type itself
            fields[key] = (value, Field(...,))
            continue

        if isinstance(value, Type[Any]):  # Error management if NO continue statement was reached.
            raise ValueError(f"Unknown field creation behavior for {key} : {value} (probably an unauthorized type keyword : ALLOWED_TYPE_KEYWORDS={ALLOWED_TYPE_KEYWORDS})")
        else:
            raise ValueError(f"Unknown field creation behavior for {key} : {value} (might be an unauthorized type : ALLOWED_TYPES={ALLOWED_TYPES} or an unknown tuple syntax, an empty list, or a list with only 1 item, ...)")
    
    return fields

def pydantic_model_from_options(**example) -> BaseModel:
    """
    INFO : You can't mix types in literals defined through this function.
    All models with literals should be created through this function so they have this property as well.
    """
    fields = extract_fields(**example)
    return create_model("-".join(example.keys()), **fields)
