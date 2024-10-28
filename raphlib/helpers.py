def escape_characters(text: str) -> str:
    escape_map = {
        '{': '{{',
        '}': '}}',
    }

    # Replace each character in escape_map with its escaped version
    for char, escaped_char in escape_map.items():
        text = text.replace(char, escaped_char)
    
    return text