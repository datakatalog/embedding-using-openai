import os
from dotenv import load_dotenv
from openai import OpenAI

# Load all .env variables
load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    raise Exception("API key not found. Make sure OPENAI_API_KEY is set in your .env file.")
client = OpenAI(api_key=api_key)
response = client.embeddings.create(
model="text-embedding-3-small",
input="Embeddings are a numerical representation of text that can be used to measure the relatedness between two pieces of text."
)
response_dict = response.model_dump()
print(response_dict)