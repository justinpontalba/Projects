# %%
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from utils import find_similar, analyze_button, session_state_to_dict, get_transformations, apply_transformations, filter_out_matches, is_dict_empty
import pickle



# %%


if __name__ == "__main__":

    st.set_page_config(layout="wide")

    session_state = st.session_state

    if 'show_new_container' not in st.session_state:
        st.session_state.show_new_container = False

    if 'analyze_clicked' not in st.session_state:
        st.session_state.analyze_clicked = False

    if 'inspect_clicked' not in st.session_state:
        st.session_state.inspect_clicked = False

    if 'apply_transformations_clicked' not in st.session_state:
        st.session_state.apply_transformations_clicked = False

    if 'is_empty_candidates' not in st.session_state:
        st.session_state.is_empty_candidates = True

    st.title("👨‍💻 Chat with your CSV")
    template= st.file_uploader("Choose a Template csv file", accept_multiple_files=False)
    transform= st.file_uploader("Choose a csv file to transform", accept_multiple_files=False)

    data_container = st.container()
    new_container = st.container()
    inspect_button_container = st.container()
    inspect_container = st.container()
    apply_transformation_container = st.container()
    transform_container = st.container()


    if template and transform: 
        session_state.df_template = pd.read_csv(template)
        session_state.df_transform = pd.read_csv(transform)

        
        with data_container:
            table1, table2, table3, table4 = st.columns(4)
            st.caption("Click the Analyze button.")
            if st.button("Analyze", type="primary"):
                with st.spinner("Analyzing..."):
                    session_state.analyze_clicked = True
            
            if session_state.analyze_clicked:
                session_state.response1, session_state.response2 = analyze_button(session_state.df_template, session_state.df_transform)
                session_state.matches, session_state.candidates = find_similar(session_state.df_template, session_state.df_transform)
                if is_dict_empty(session_state.candidates):
                    st.session_state.is_empty_candidates = True
                else:
                    st.session_state.is_empty_candidates = False

                with table4:
                    if type(session_state.response1) == str:
                        st.write(session_state.response1)
                    else:  
                        for i,j in zip(session_state.response1.keys(), session_state.response1.values()):
                            st.write(f"**{i}**:{j}\n")
                with table2:
                    if type(session_state.response2) == str:
                        st.write(session_state.response2)
                    else:
                        for k,l in zip(session_state.response2.keys(), session_state.response2.values()):
                            st.write(f"**{k}**:{l}\n")
                        
                        session_state.show_new_container = True


            with table1:

                st.caption("Table to Transform")
                with st.spinner():
                    st.dataframe(session_state.df_transform)

            with table3:
                st.caption("Template Table")
                with st.spinner():
                    st.dataframe(session_state.df_template)
        
        with new_container:
            st.title("Transforming Columns")
            
            table1, table2 = st.columns(2)
            if session_state.show_new_container:
                st.caption("The column on the left do not require user intervention. The column on the right requires you to confirm which column the transformations will be applied to.")
                with table1:
                    for i,j in zip(session_state.matches.keys(), session_state.matches.values()):
                        if len(j) > 1:
                            out_msg = f"Columns **{j}** are identical and will be mapped to: **{i}** in the TEMPLATE. Only 1 column will be maintained after the transformation. **Reason:** These columns are semantically similar."
                            st.write(out_msg)
                        
                        else:
                            out_msg = f"Column: **{j}** will be mapped to: **{i}** in the TEMPLATE. **Reason:** These columns are semantically similar."
                            st.write(out_msg)
                
                with table2:
                    if not session_state.is_empty_candidates:
                        # Define column names for the empty dataframe
                        columns = list(session_state.candidates.keys())
                        # Create an empty dataframe with the defined columns
                        empty_df = pd.DataFrame(columns=columns)

                        # Check if the session state variable is already defined
                        if "df" not in session_state:
                            # Assign the initial data to the session state variable
                            session_state.df = empty_df
                            session_state.row = pd.Series(index=columns)

                        # Create a selectbox for each column in the current row
                        for col in columns:
                            # Get unique values from the corresponding column in the resource_data dataframe
                            
                            values = session_state.candidates[col]

                            out_msg = f"Columns **{values}** require user validation prior to mapping. The choice between **{values}** will be mapped to **{col}** in the TEMPLATE"
                            st.write(out_msg)

                            # Create a selectbox for the current column and add the selected value to the current row
                            index = values.index(session_state.row[col]) if session_state.row[col] in values else 0

                            session_state.row[col] = st.selectbox(col, values, key=col, index=index)
        
        with inspect_button_container:
            st.caption("Click the Inspection button")
            if st.button("Inspect", type="primary"):
                with st.spinner(""):
                    session_state.inspect_clicked = True

        with inspect_container:
            st.title("Inspecting Transformations")
            col1,col2,col3 = st.columns(3)
            if session_state.inspect_clicked:
                st.caption(f"Use the editable dataframe to alter the transformations as needed. The transformation instructions will be sent to the OpenAI Completion API end-point for modification. For columns that do not require transformation after inspection, fill the **Transformation Description** column with **n/a**")
                if not session_state.is_empty_candidates:
                    session_state.transform_dict = session_state_to_dict(session_state, session_state.matches, session_state.candidates)
                    session_state.transformation_df = get_transformations(session_state.transform_dict, session_state.df_transform, session_state.df_template)
                    with col2:
                        session_state.edited_df = st.data_editor(session_state.transformation_df)
                else:
                    st.caption("Column contents are identical and do not require transformation")
                    df_no_transform = filter_out_matches(session_state.matches, session_state.df_transform)
                    st.dataframe(df_no_transform)

        
        if not session_state.is_empty_candidates:
            with apply_transformation_container:
                st.caption("Click Apply Transformatons button")
                if st.button("Apply Transformatons", type = "primary"):
                    with st.spinner(""):
                        session_state.apply_transformations_clicked= True
                        session_state.transformed_df = apply_transformations(session_state.edited_df, session_state.df_transform, session_state.df_template)
            
            with transform_container:
                col1,col2,col3 = st.columns(3)
                with col2:
                    if session_state.apply_transformations_clicked:
                        st.dataframe(session_state.transformed_df)
                
                
        


                    
                    





            





                