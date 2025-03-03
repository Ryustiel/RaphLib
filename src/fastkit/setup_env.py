"""
A quick and easy way to access a ready to go LLM.
"""
import os
import dotenv

from langchain_openai import AzureChatOpenAI

def setup_env():
    """
    blah
    """
    # TODO : Make this a general function that just checks for environment variables, then return their values
    # TODO : This module should refresh env when imported
    # TODO : Update the actual models in chat_models, while importing them and exposing them when they're first invoked
    # => Use a module was container class that maintains references to instances of chat models generated through here

    VARIABLES = (
        "OPENAI_API_TYPE",
        "OPENAI_API_VERSION",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
    )

    dotenv.load_dotenv()

    missing = set()

    for variable in VARIABLES:
        value = os.getenv(variable)
        if value is None:
            missing.add(variable)
        else:
            os.environ[variable] = value

    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

def get_llm(deployment_name: str) -> AzureChatOpenAI:

    setup_env()

    return AzureChatOpenAI(
        deployment_name=deployment_name,
        temperature=0.4,
        max_tokens=500,
    )