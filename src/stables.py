"""
Stable models are pydantic models that are designed to parse themselves into a "closest" relative on model_validate, 
nulling all missing fields.
"""
import re
import json
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

from .helpers import get_all_fields_as_optional

# ================================================================= REGEX FOR STABLE MODELS =================================================================

RE_PATTERN_LIST_INCOMPLETE = re.compile(
    r'\[[^{}\]]*,\s*"[^"]*"?$' # Match any list pattern potentially missing a bracket or a quote
)

RE_PATTERN_KEY_INCOMPELTE = re.compile(
    r'([\[{,]\s*"[^{}\[\]\"]*"?\s*:?\s*{?\s*}?\s*$)' # Match any {[, followed by a "key": like structure without a value. These patterns will be removed.
)

# ================================================================= STABLE MODEL =================================

class StableModel(BaseModel):
    """
    A StableModel is a pydantic model that has a defined structure and can be parsed into a "closest" relative,
    usually with None values for the missing fields.
    Stable models can handle partial json objects.
    """
    is_stable_model_complete: bool = False
    
    """
    Idea : 
    1. Just remove dirty parts of the incoming json and add as many closing brackets as needed.
    2. Parse a "fully optional" version of the model.
    3. Once the stream input dries out, attempts to parse the model with the actual object (that has mandatory fields) and set [is_complete], 
    raise errors if it does not match. (Stable validation error)
    4. Delta mode to return a version of this object that only has the latest updated chunk, but with the right field hierarchy.
    """

    def extract_delta(self, partial_json: str) -> 'StableModel':
        """
        Will produce a StableModel that only has the difference between the current content and the new content.
        All the other fields will be set to None.
        """
        new_model = self.from_partial_json(partial_json)
        # 2. Compute differences between the new_model and the current_model (both should be json)
        # 3. Create a 3rd model with all the values set to None but the one(s) who were updated, = to the textdelta exactly
        # Behavior specifications : For all values that are not exactly str types, don't include them unless the value can be parsed in the proper type.
        # This behavior should be replicated on the from_partial_json() class method so that the difference stuff can still happen.
        ...


    @classmethod
    def from_partial_json(cls, partial_json: str) -> 'StableModel':
        """
        Parse a partial input JSON string and return an instance of StableModel.

        stream_list_items (bool) : Whether or not to stream list items. If set to False list items will only appear when completed.
        
        Steps:
        1. Remove the trailing comma if there is one at the end of the string (potentially separated by a space character)
        2. If the string ends with a key that is not fully written (like {"ke or {"abc": "123", "de) then remove it.
        3. If the string ends with a valid key and a value that's not fully written like {"abc":"12 then simply add the missing "
        4. Note the openings {[... that do not have a matching closing, and complete the json by adding the correct }]... in order
        5. Parse the repaired JSON with Pydantic.
        """
        repaired_json = partial_json

        if len(repaired_json) < 1:
            return ""  # FIXME : Use Default empty model
        
        else:
            # Step 1: Remove any json``` pattern that is recurrent on those outputs. This is only for json outputs.
            if repaired_json[0] == '`':
                if len(repaired_json) < 7:
                    return ""  # FIXME : Use Default empty model
                else:
                    repaired_json = repaired_json[7:]  # Cut off the json``` structure

            # Step 2: Check for incomplete list patterns like ["abc", "12 and enclose them with a quote.
            match = RE_PATTERN_LIST_INCOMPLETE.search(repaired_json)
            if match:
                # Add an extra quote if the last character of the list structure was not a "
                if repaired_json[match.end() - 1] != "\"":
                    repaired_json += "\""

            elif len(repaired_json) >= 2 and repaired_json[-2:] == "[\"":  # Step 2b: Specifically handles the pattern [" (no comma)
                repaired_json = repaired_json[:-1]  # Remove lone quote

            else:
                # Step 3: Remove any incomplete key value pattern
                match = RE_PATTERN_KEY_INCOMPELTE.search(repaired_json)
                if match:
                    # Remove from the last comma and quote onwards
                    repaired_json = repaired_json[:match.start() + 1]

            # Step 4: Remove trailing comma or comma quote
            comma_pattern = re.compile(r'(,\s*"?$)')
            match = comma_pattern.search(repaired_json)
            if match:
                repaired_json = repaired_json[:match.start()]

            # One last check after all the cutting have been done
            if len(repaired_json) < 4:
                return ""  # FIXME : Use Default empty model

            # Step 5: Count the quotes. If uneven then add a quote at the end of the string
            comma_count = repaired_json.count("\"")
            if comma_count % 2 == 1:
                repaired_json += "\""

            # Step 6: Balance the brackets by adding closing brackets
            # Count opening and closing braces/brackets
            stack = []
            for char in repaired_json:
                if char in '{[':
                    stack.append(char)
                elif char in '}]':
                    if stack:
                        last = stack[-1]
                        if (last == '{' and char == '}') or (last == '[' and char == ']'):
                            stack.pop()
                        else:
                            # Mismatched bracket, ignoring for simplicity
                            pass

            # Add the necessary closing brackets in reverse order
            closing_brackets = {'{': '}', '[': ']'}
            while stack:
                opening = stack.pop()
                repaired_json += closing_brackets.get(opening, '')

            # Step 7: Remove the closing `
            while len(repaired_json) > 0 and repaired_json[-1] == "`":  # popping ` out
                repaired_json = repaired_json[:-1]

            try:
                repaired_dict = json.loads(repaired_json)
                return json.dumps(repaired_dict, indent=4)
            except ValidationError as e:
                raise e
       
    @classmethod
    def from_model(cls, model: Type[BaseModel]) -> Type['StableModel']:
        """
        Build a stable model class from a reference pydantic model.
        """
        # 1. Recursively checks all of the attributes
        # 2. Makes them all optional if they are not already
        # 3. Add validation specifications according to the following : 
        # Behavior specifications : For all values that are not exactly str types, don't include them unless the value can be parsed in the proper type.
        # This behavior should be replicated on the from_partial_json() class method so that the difference stuff can still happen
        # 4. Returns the new StableModel class.
        ...

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
