import os
import dotenv

def setup_env():

    dotenv.load_dotenv()

    # Set up your Azure OpenAI credentials
    os.environ["OPENAI_API_TYPE"] = os.getenv("OPENAI_API_TYPE")
    os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
    os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
