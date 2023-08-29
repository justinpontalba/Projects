from langchain import OpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
import json

# Setting up the api key
import environ

env = environ.Env()
environ.Env.read_env()

API_KEY = env("apikey")

# Save JSON object to a file
def save_json_file(filename, data):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

# Load JSON object from a file
def load_json_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def decode_response(response: str) -> dict:
    """This function converts the string response from the model to a dictionary object.

    Args:
        response (str): response from the model

    Returns:
        dict: dictionary with response data
    """
    return json.loads(response)

def create_and_query_agent(df, prompt, query, path):

    output = load_json_file(path)
    
    # llm = OpenAI(temperature=0, openai_api_key=API_KEY)
    # agent = create_pandas_dataframe_agent(llm, df, verbose=False)

    # Run the prompt through the agent.
    # response = agent.run(prompt + query)

    # try:
    #     output = decode_response(response.__str__())

    # except Exception:
    #     output = response.__str__()

    # Convert the response to a string.
    return output