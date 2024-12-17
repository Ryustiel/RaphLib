"""
A quick and easy way to access a ready to go LLM.
"""
import os
import dotenv

from langchain_openai import AzureChatOpenAI

def setup_env():

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