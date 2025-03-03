
import os
import dotenv
dotenv.load_dotenv(override=True)

from langchain_openai import ChatOpenAI, AzureChatOpenAI

O3_MINI = ChatOpenAI(
    api_key = os.environ["OPENAI_API_KEY"],
    model= "o3-mini",
)

GPT_4O_MINI = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["OPENAI_API_VERSION"],

    deployment_name="gpt-4o-mini",
    temperature=0.4,
    max_tokens=500,
)

GPT_4O = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["OPENAI_API_VERSION"],

    deployment_name="gpt-4o",
    temperature=0.4,
    max_tokens=500,
)
