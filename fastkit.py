"""
A quick and easy way to access a ready to go LLM.
"""
from raphlib import setup_env
from langchain_openai import AzureChatOpenAI

setup_env()

def get_llm(deployment_name: str) -> AzureChatOpenAI:

    return AzureChatOpenAI(
        deployment_name=deployment_name,
        temperature=0.4,
        max_tokens=500,
    )