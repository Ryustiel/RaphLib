ESCAPE_MAP = {
        '{': '{{',
        '}': '}}',
    }

def escape_characters(text: str) -> str:
    
    # Replace each character in escape_map with its escaped version
    for char, escaped_char in ESCAPE_MAP.items():
        text = text.replace(char, escaped_char)
    
    return text