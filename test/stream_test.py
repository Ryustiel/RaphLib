from pydantic import BaseModel
from raphlib.fastkit import get_llm

LLM = get_llm('gpt-4o')

from raphlib import BaseTool, LLMWithTools, StreamEvent, AITextResponseChunk, pydantic_model_from_options
from typing import Type

class TemplateTool(BaseTool):
    name: str = "get_information"
    description: str = """This tool must be run once before replying to the user."""
    args_schema: Type[BaseModel] = pydantic_model_from_options(
        random_fruit_name=str
    )
    async def _arun(self, inp: BaseModel) -> str:
        return f"Now respond to the user with mentioning bananas, {inp.random_fruit_name} and coconut."

TESTLLM = LLMWithTools(LLM, [TemplateTool()])

for event in TESTLLM.stream("Tell me about the information from the tool, pick pear."):
    if isinstance(event, AITextResponseChunk):
        print(event.content)
    elif not isinstance(event, StreamEvent):
        print("\n\nRESPONSE\n\n", event.content)
    else:
        print(type(event).__name__)
        