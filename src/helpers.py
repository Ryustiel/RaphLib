import re
import time
import asyncio
from typing import List, Optional, Union, Any, Awaitable, Type, Dict, Any, get_args, get_origin
from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

ESCAPE_MAP = {
        '{': '{{',
        '}': '}}',
    }


# ================================================================= ESCAPE CHARACTERS

def escape_characters(text: str) -> str:

    # Replace each character in escape_map with its escaped version
    for char, escaped_char in ESCAPE_MAP.items():
        text = text.replace(char, escaped_char)
    
    return text


# ================================================================= BALANCE RESULTS

def balance_results(
    resources: List[List],
    max_total: int,
    max_per_list: Optional[Union[int, List[int]]] = None
) -> List:
    """
    Balances and combines top results from multiple resource lists.

    Args:
        resources (List[List]): A list of lists, where each sublist contains ordered results from a different source.
        max_total (int): The maximum total number of results to include in the final list.
        max_per_list (Optional[Union[int, List[int]]]):
            - If an integer is provided, it sets the same maximum number of results to take from each list.
            - If a list is provided, each element specifies the maximum number of results to take from the corresponding resource list.
            - If None, no per-list limits are enforced beyond `max_total`.

    Returns:
        List: A single list containing the balanced top results from each input list.

    Raises:
        ValueError: If `max_per_list` is a list but its length does not match the number of resource lists.
    """
    # Validate max_per_list
    if isinstance(max_per_list, list):
        if len(max_per_list) != len(resources):
            raise ValueError(
                "Length of max_per_list mask must match the number of resource lists."
            )
        limits = max_per_list
    elif isinstance(max_per_list, int):
        limits = [max_per_list] * len(resources)
    elif max_per_list is None:
        # If no max_per_list is provided, set to infinity for all lists
        limits = [float('inf')] * len(resources)
    else:
        raise TypeError(
            "max_per_list must be either an integer, a list of integers, or None."
        )

    # Make copies of the resource lists to avoid modifying the original inputs
    copies = [lst.copy() for lst in resources]

    # Initialize the result list
    result = []

    # Initialize a list to keep track of how many items have been taken from each resource
    counts = [0] * len(copies)

    # Continue iterating until the max_total is reached
    while len(result) < max_total:
        made_progress = False  # Flag to check if any item was added in this iteration

        # Iterate over each copy to take one item at a time
        for i, lst in enumerate(copies):
            # Check if we've already taken the maximum allowed from this list
            if counts[i] >= limits[i]:
                continue

            # Check if the current list has any items left
            if not lst:
                continue

            # Take the top (first) item from the current list
            item = lst.pop(0)
            result.append(item)
            counts[i] += 1
            made_progress = True  # An item was successfully added

            # If we've reached the max_total, exit early
            if len(result) >= max_total:
                break

        # If no items were added in this pass, no further items can be taken
        if not made_progress:
            break

    return result


# ================================================================= FIRST TASK COMPLETED

async def first_completed(*tasks: Awaitable[Any]) -> Any:
    """
    Run multiple awaitables concurrently and return the result of the first one
    that completes. Cancel all other awaitables once one completes.

    :param tasks: Variable number of awaitable objects (coroutines, Tasks, Futures)
    :return: The result of the first completed awaitable
    :raises: Exception raised by the first completed awaitable
    """
    if not tasks:
        raise ValueError("At least one awaitable must be provided")

    # Create tasks from the awaitables
    task_list: List[asyncio.Task] = [asyncio.create_task(task) for task in tasks]

    try:
        # Wait until the first task completes
        done, pending = await asyncio.wait(
            task_list, return_when=asyncio.FIRST_COMPLETED
        )
        
        # Retrieve the first completed task
        first_done = done.pop()

        # Cancel all pending tasks
        for pending_task in pending:
            pending_task.cancel()
        
        # Optionally, wait for the cancellation to finish
        await asyncio.gather(*pending, return_exceptions=True)

        # Return the result of the first completed task
        return first_done.result()

    except Exception as e:
        # If the first completed task raised an exception, re-raise it
        raise e

    finally:
        # Ensure all tasks are cleaned up
        for task in task_list:
            if not task.done():
                task.cancel()
        await asyncio.gather(*task_list, return_exceptions=True)


# ================================================================= MAKE FIELDS OPTIONAL

def optionalize_type(type_: Any) -> Any:
    """
    Recursively make a type annotation optional. Handles BaseModel subclasses,
    lists, and other generic types.
    """
    origin = get_origin(type_)
    if origin is not None:
        # Handle generic types such as List, Dict, Set, etc.
        args = get_args(type_)
        optionalized_args = tuple(optionalize_type(arg) for arg in args)
        return Optional[origin[optionalized_args]]
    elif isinstance(type_, type) and issubclass(type_, BaseModel):
        # Handle nested Pydantic models recursively
        nested_fields = get_all_fields_as_optional(type_)
        nested_model = create_model(
            type_.__name__,
            **nested_fields
        )
        return Optional[nested_model]
    else:
        # Base case: For simple types, just wrap with Optional
        return Optional[type_]

def get_all_fields_as_optional(model: Type[BaseModel]) -> Dict[str, FieldInfo]:
    """
    Return the model fields with all fields set to optional with None as the default.
    If a field is another Pydantic model or a container, process it recursively.
    """
    fields: Dict[str, Any] = {}
    for name, field in model.model_fields.items():
        # Recursively make the field type optional
        optional_type = optionalize_type(field.annotation)
        # Assign a default value of None to the field
        fields[name] = (optional_type, None)
    
    return fields

# ================================================================= REPAIR JSON


RE_PATTERN_TRAILING_VALUE = re.compile(
    r'[:,]\s*(?!(["]))[+-]?[\d.eE]*[a-zA-Z]*$'  # Match any ,: followed by a value without a " that was not enclosed by a bracket or any other closing statement or stuff
)

RE_PATTERN_LIST_INCOMPLETE = re.compile(
    r'\[[^{}\]]*,\s*"[^"]*"?$' # Match any list pattern potentially missing a bracket or a quote
)

RE_PATTERN_KEY_INCOMPELTE = re.compile(
    r'([\[{,]\s*"[^{}\[\]\"]*"?\s*:?\s*{?\s*}?\s*$)' # Match any {[, followed by a "key": like structure without a value. These patterns will be removed.
)

RE_PATTERN_GENERAL_LIST_FIX = re.compile(
    r'[\[,](\s*"[^,"]*)$|([\[,]\s*\d*.)$'
)

EMPTY_JSON = "{}"

def repair_list(trucated_list: str) -> str:
    """
    Fix any list, even lists of digits [1, 2, ...] and mixed lists [1, "2", ...]
    """
    if len(trucated_list) < 2:
        return "[]"
    elif trucated_list[-1] != "]":
        match = RE_PATTERN_GENERAL_LIST_FIX.search(trucated_list)
        if match:
            trucated_list = trucated_list[:match.start()]
            if len(trucated_list) < 2:
                return "[]"
        return trucated_list + "]"
    else:
        return trucated_list


def repair_json(truncated_json: str) -> str:
        """
        Repair a trucated version of a valid JSON string.
        
        Steps:
        1. 
        """
        repaired_json = truncated_json

        if len(repaired_json) < 1:
            return EMPTY_JSON
        
        else:
            # Step 1: Remove any json``` pattern that is recurrent on those outputs. This is only for json outputs.
            if repaired_json[0] == '`':
                if len(repaired_json) < 8:
                    return EMPTY_JSON
                else:
                    repaired_json = repaired_json[7:]  # Cut off the json``` structure
            
            if repaired_json[-1] == '.':
                repaired_json  = repaired_json[:-1]

            # Step 2: Remove any trailing non string value that might be incomplete
            # These are digits strings (including float points) or incomplete keywords such as bool or null (that don't start with a comma)
            # Patterns look like : colon or , comma followed by \s* and digits or letters with no opening " quote in between
            # Removal of dict values this way will let the associated key be removed properly in step 4
            match = RE_PATTERN_TRAILING_VALUE.search(repaired_json)
            if match:
                repaired_json = repaired_json[:match.start()]  # Remove the trailing non string value

            # Step 3: Check for incomplete list patterns like ["abc", "12 and enclose them with a quote.
            match = RE_PATTERN_LIST_INCOMPLETE.search(repaired_json)
            if match:
                # Add an extra quote if the last character of the list structure was not a "
                if repaired_json[match.end() - 1] != "\"":
                    repaired_json += "\""

            elif len(repaired_json) >= 2 and repaired_json[-2:] == "[\"":  # Step 2b: Specifically handles the pattern [" (no comma)
                repaired_json = repaired_json[:-1]  # Remove lone quote

            else:
                # Step 4: Remove any incomplete key value pattern
                match = RE_PATTERN_KEY_INCOMPELTE.search(repaired_json)
                if match:
                    # Remove from the last comma and quote onwards
                    repaired_json = repaired_json[:match.start() + 1]

            # Step 5: Remove trailing comma or comma quote
            comma_pattern = re.compile(r'(,\s*"?$)')
            match = comma_pattern.search(repaired_json)
            if match:
                repaired_json = repaired_json[:match.start()]

            # One last check after all the cutting have been done
            if len(repaired_json) < 4:
                return EMPTY_JSON

            # Step 6: Count the quotes. If uneven then add a quote at the end of the string
            comma_count = repaired_json.count("\"")
            if comma_count % 2 == 1:
                repaired_json += "\""

            # Step 7: Balance the brackets by adding closing brackets
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

            # Step 8: Remove the closing `
            while len(repaired_json) > 0 and repaired_json[-1] == "`":  # popping ` out
                repaired_json = repaired_json[:-1]

            return repaired_json
            
# FIXME : Compute diff string is cutting one character off for some reason, only in list parsed strings. To be fixed.
def diff_dict(exclude_dict: dict, data_dict: dict, compute_diff_string: bool= True) -> dict:
    """
    Compute the difference between two dictionaries.
    Only the data_dict's items that are not in the included_dict will be kept.

    If compute_diff_string is True, only the difference between two strings will be provided instead of the whole string items.

    NOTE : The dictionary should not contain a {"key": [[]]} structure with a double list.
    Any dict generated from the parsing of a json is however compatible.
    """
    result = {}
    for datakey, datavalue in data_dict.items():
        filtervalue = exclude_dict.get(datakey)  # Get filtervalue or None if key is not in exclude_dict
        if isinstance(filtervalue, dict) and isinstance(datavalue, dict):  # Nested dictionaries
            diff = diff_dict(exclude_dict=filtervalue, data_dict=datavalue, compute_diff_string=compute_diff_string)
            if diff:  # Only include if there are differences
                result[datakey] = diff
        elif isinstance(filtervalue, list) and isinstance(datavalue, list):  # If both are lists
            # Compute list difference by including items in datavalue that are not in filtervalue
            diff_list = []
            # Extend the values lists so that the zipping works
            if len(filtervalue) < len(datavalue):
                for _ in datavalue: filtervalue.append(None)
            if len(datavalue) < len(filtervalue):
                for _ in filtervalue: datavalue.append(None)

            for excludeitem, dataitem in zip(filtervalue, datavalue):  # Value should come in the same order
                if isinstance(excludeitem, dict) and isinstance(dataitem, dict):
                    subdictionaries_diff = diff_dict(exclude_dict=excludeitem, data_dict=dataitem, compute_diff_string=compute_diff_string)
                    if subdictionaries_diff:
                        diff_list.append(subdictionaries_diff)
                # Cannot be a list in a list in a json
                elif excludeitem != dataitem and excludeitem is not None:
                    if compute_diff_string and isinstance(excludeitem, str) and isinstance(dataitem, str):
                        if len(excludeitem) < len(dataitem):
                            diff_list.append(dataitem[len(excludeitem):])
                    else:
                        diff_list.append(dataitem)
                    
            if diff_list:  # Only include if there are differences
                result[datakey] = diff_list
        elif compute_diff_string and isinstance(filtervalue, str) and isinstance(datavalue, str):
            if len(filtervalue) < len(datavalue):
                result[datakey] = datavalue[len(filtervalue):]  # Return str diff
        elif filtervalue != datavalue:  # If the values differ or key is only in dict2
            result[datakey] = datavalue
    return result

# ================================================================= LAPTIMER

class LapTimer:
    def __init__(self):
        """
        Initializes the LapTimer by storing the current high-resolution time.
        """
        self.start_time = time.perf_counter()
        self.last_lap_time = self.start_time
        self.is_first_lap = True

    def lap(self, message: Optional[str] = None):
        """
        Calculates and prints the time difference between the last lap and the current lap.
        If it's the first lap, it calculates the difference from initialization.
        """
        current_time = time.perf_counter()
        time_diff = current_time - self.last_lap_time
        
        # Update the last lap time to the current time
        self.last_lap_time = current_time
        
        # Calculate seconds and milliseconds
        seconds = int(time_diff)
        milliseconds = int((time_diff - seconds) * 1000)
        
        print("")

        if message:
            print(message, ":", end=" ")

        if self.is_first_lap:
            print(f"Time since initialization: {seconds} seconds {milliseconds} milliseconds")
            self.is_first_lap = False
        else:
            print(f"Time since last lap: {seconds} seconds {milliseconds} milliseconds")

        print("\n")


# ================================================================= MAIN

if __name__ == "__main__":
    # Sample data from different search engines
    resources = [
        ["Google_result1", "Google_result2", "Google_result3"],
        ["Bing_result1", "Bing_result2"],
        ["DuckDuckGo_result1", "DuckDuckGo_result2", "DuckDuckGo_result3", "DuckDuckGo_result4"]
    ]

    max_total = 5
    max_per_list = 2  # Maximum 2 results from each list

    balanced = balance_results(resources, max_total, max_per_list)
    print(balanced)

    # Sample data from different search engines
    resources = [
        ["Google_result1", "Google_result2", "Google_result3"],
        ["Bing_result1", "Bing_result2"],
        ["DuckDuckGo_result1", "DuckDuckGo_result2", "DuckDuckGo_result3", "DuckDuckGo_result4"]
    ]

    max_total = 7
    max_per_list_mask = [1, 2, 3]  # Google:1, Bing:2, DuckDuckGo:3

    balanced = balance_results(resources, max_total, max_per_list_mask)
    print(balanced)