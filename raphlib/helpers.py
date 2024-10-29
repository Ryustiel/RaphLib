from typing import List, Optional, Union

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