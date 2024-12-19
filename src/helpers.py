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