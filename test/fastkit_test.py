from raphlib import LLMFunction
from raphlib.fastkit import get_llm
from pydantic import BaseModel
from typing import List

LLM = get_llm('gpt-4o')

class Explanation(BaseModel):
    word: str
    explanation: str

response = LLMFunction(
    LLM,
    """
    Explain the different words in these lyrics as {{\"explanations\":[{{\"word\":\"explanation\"}}]}}.
    Lyrics : {lyrics}
    """,
    explanations=List[Explanation]
).invoke({"lyrics": "I like cats, do you like spaghetti?"})

print(response)
