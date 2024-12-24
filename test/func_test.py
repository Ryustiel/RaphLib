import time

from pydantic import BaseModel
from typing import List, Type

from raphlib import LLMFunction, StableModel, BaseTool, LLMWithTools, pydantic_model_from_options
from raphlib.fastkit import get_llm

LLM = get_llm('gpt-4o')

class Word(BaseModel):
    root: str
    affixes: str

class Explanation(BaseModel):
    word: Word
    three_random_float: List[float]
    random_list_of_int: List[int]
    explanation: str
    synonyms: List[str]
    example_sentences: List[str]

class Explanations(BaseModel):
    explanations: List[Explanation]

# Define a Tool

class TemplateTool(BaseTool):
    name: str = "get_information"
    description: str = """Provide information about a vegetable or fruit."""
    args_schema: Type[BaseModel] = pydantic_model_from_options(
        vegetable=str
    )
    async def _arun(self, inp: BaseModel) -> str:
        return f"Potatoes are synonymous with the literal word \"balabala\". You have to list this as a synonym of potato in your response."

LLM_WITH_POTATO_TOOL = LLMWithTools(LLM, [TemplateTool()])

func = LLMFunction(LLM_WITH_POTATO_TOOL,
    """
    Explain the different words in these lyrics as per the format instructions.
    Lyrics : {lyrics}
    """,
    pydantic_model=Explanations,
)

for event in func.stream({"lyrics": "I like potatoes"}):
    if isinstance(event, StableModel):
        print("\n"*10, event.model_dump(exclude_none=True))
    else:
        print(type(event).__name__)
    time.sleep(0.001)  # Make the output more readable by adding a delay

# => Faire des changements sur MeepInterface qui sera lui même une instance de tool (hérite de raphlib.stream.BaseTool)
# => Puis connecter discord
# => Et ajouter le support des flags
# => Faire fonctionner les menus
# => Puis ajouter le système de références dans MeepChatHistory
# => Puis Implémenter la streaming interface dans Meep et faire que ça marche avec un nouveau tool ou endpoint dans discord
# => Puis reprendre le travail sur la mémoire en réparant le bug
