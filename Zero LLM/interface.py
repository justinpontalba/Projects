# %%
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from utils import find_similar, analyze_button
# %%



if __name__ == "__main__":

    st.set_page_config(layout="wide")


    st.title("👨‍💻 Chat with your CSV")
    template= st.file_uploader("Choose a Template csv file", accept_multiple_files=False)
    transform= st.file_uploader("Choose a csv file to transform", accept_multiple_files=False)
    show_new_container = False

    data_container = st.container()
    new_container = st.container()

    if template and transform: 
        df_template = pd.read_csv(template)
        df_transform = pd.read_csv(transform)

        
        with data_container:
            table1, table2, table3, table4 = st.columns(4)
            # if st.button("Analyze", type="primary"):
            #     with st.spinner("Analyzing..."):
            response1, response2 = analyze_button(df_template, df_transform)
            matches, candidates = find_similar(df_template, df_transform)
            with table4:
                if type(response1) == str:
                    st.write(response1)
                else:  
                    for i,j in zip(response1.keys(), response1.values()):
                        st.write(f"**{i}**:{j}\n")
            with table2:
                if type(response2) == str:
                    st.write(response2)
                else:
                    for k,l in zip(response2.keys(), response2.values()):
                        st.write(f"**{k}**:{l}\n")
            show_new_container = True

            with table1:
                # selected_columns = st.multiselect("Select columns", df_transform.columns)
                st.caption("Table to Transform")
                with st.spinner():
                    st.dataframe(df_transform)
                # if selected_columns:
                #     selected_df = df_transform.copy()
                #     selected_df = selected_df[selected_columns]
                #     st.write("Selected Columns:")
                #     st.dataframe(selected_df)
            with table3:
                st.caption("Template Table")
                with st.spinner():
                    st.dataframe(df_template)
        
        with new_container:
            st.title("Analyzing and Transforming Columns")
            table1, table2 = st.columns(2)
            if show_new_container:
                with table1:
                    for i,j in zip(matches.keys(), matches.values()):
                        if len(j) > 1:
                            out_msg = f"Columns **{j}** are identical and will be mapped to: **{i}** in the TEMPLATE. Only 1 column will be maintained after the transformation."
                            st.write(out_msg)
                        
                        else:
                            out_msg = f"Column: **{j}** will be mapped to: **{i}** in the TEMPLATE."
                            st.write(out_msg)
                
                with table2:
                    # Define column names for the empty dataframe
                    columns = list(candidates.keys())
                    # Create an empty dataframe with the defined columns
                    empty_df = pd.DataFrame(columns=columns)

                    session_state = st.session_state
                    # Check if the session state variable is already defined
                    if "df" not in session_state:
                        # Assign the initial data to the session state variable
                        session_state.df = empty_df
                        session_state.row = pd.Series(index=columns)

                    # Create a selectbox for each column in the current row
                    for col in columns:
                        # Get unique values from the corresponding column in the resource_data dataframe
                        
                        values = candidates[col]

                        out_msg = f"Columns **{values}** require user validation prior to mapping. The choice between **{values}** will be mapped to **{col}** in the TEMPLATE"
                        st.write(out_msg)

                        # Create a selectbox for the current column and add the selected value to the current row
                        index = values.index(session_state.row[col]) if session_state.row[col] in values else 0

                        session_state.row[col] = st.selectbox(col, values, key=col, index=index)
                    
                    print(session_state)
                            # options = {}
                            # for key,values in candidates.items():
                            #     out_msg = f"Columns **{values}** require user validation prior to mapping. The choice between **{values}** will be mapped to **{key}** in the TEMPLATE"
                            #     st.write(out_msg)
                            #     # option = st.selectbox(f"Template: **{i}**", j)
                            #     options[i] = option
                    





            





                