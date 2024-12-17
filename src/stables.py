"""
Stable models are pydantic models that are designed to parse themselves into a "closest" relative.
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
    Type, 
    Optional, 
    get_args,
    get_origin,
)
from pydantic import BaseModel, Field, create_model
from pydantic_core import ValidationError

# ================================================================= STABLE MODEL =================================

class StableField(BaseModel):
    """
    Holds partial values and ways to evaluate them.
    Lets you retrieve the last "partial value" or "textdelta" that was used to populate a particular field.
    """
    is_complete: bool = False
    type: Type  # The expected type of the field when the value is complete.
    value: str = ""
    delta: Optional[str] = None
    ...

class StableModel(BaseModel):
    """
    A StableModel is a pydantic model that has a defined structure and can be parsed into a "closest" relative,
    usually with None values for the missing fields.
    Stable models can handle partial json objects.
    """
    is_complete: bool = False
    ...

    """
    Idea : 
    1. Just remove dirty parts of the incoming json and add as many closing brackets as needed.
    2. Parse a "fully optional" version of the model.
    3. Once the stream input dries out, attempts to parse the model with the actual object (that has mandatory fields) and set [is_complete], 
    raise errors if it does not match. (Stable validation error)
    4. Support detecting the "most partial field", so that maybe we don't need that StableField object...
    Optionally compute a chunk diff with the previous output (stored) and populate StableField with the latest bits of chunks when asked for. 
    """

# ================================================================= CREATE MODEL =================================

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
