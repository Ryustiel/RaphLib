from pydantic import BaseModel
from typing import List

from raphlib import LLMFunction
from raphlib.fastkit import get_llm

LLM = get_llm('gpt-4o')

class Explanation(BaseModel):
    word: str
    explanation: str

class Explanations(BaseModel):
    explanations: List[Explanation]

response = LLMFunction(
    LLM,
    """
    Explain the different words in these lyrics as per the format instructions.
    Lyrics : {lyrics}
    """,
    pydantic_model=Explanations,
).invoke({"lyrics": "I like cats, do you like spaghetti?"})

print(response)
