# %%
import streamlit as st
import pandas as pd
import numpy as np
import json
from agent import query_agent, create_agent
from io import StringIO
from utils import find_similar

# %%
def decode_response(response: str) -> dict:
    """This function converts the string response from the model to a dictionary object.

    Args:
        response (str): response from the model

    Returns:
        dict: dictionary with response data
    """
    return json.loads(response)

def write_response(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """

    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.bar_chart(df)

    # Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.line_chart(df)

    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)

if __name__ == "__main__":

    st.set_page_config(layout="wide")

    st.title("👨‍💻 Chat with your CSV")
    template= st.file_uploader("Choose a Template csv file", accept_multiple_files=False)
    transform= st.file_uploader("Choose a csv file to transform", accept_multiple_files=False)

    data_container = st.container()

    if template and transform: 
        df_template = pd.read_csv(template)
        df_transform = pd.read_csv(transform)

        
        with data_container:
            table1, table2 = st.columns(2)
            with table1:
                # selected_columns = st.multiselect("Select columns", df_transform.columns)
                st.caption("Table to Transform")
                st.dataframe(df_transform)
                # if selected_columns:
                #     selected_df = df_transform.copy()
                #     selected_df = selected_df[selected_columns]
                #     st.write("Selected Columns:")
                #     st.dataframe(selected_df)
            with table2:
                st.caption("Template Table")
                st.dataframe(df_template)
        if st.button("Analyze Table", type="primary"):
            matches, candidates = find_similar(df_template, df_transform)
            st.json(matches)
            st.json(candidates)