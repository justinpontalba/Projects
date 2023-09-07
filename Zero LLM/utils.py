
# %%
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from agent import create_and_query_agent
import ast


# %%
def drop_keys_and_values(target_dict, reference_dict):
    print(target_dict)
    print(reference_dict)
    keys_to_drop = []
    for key, values in target_dict.items():
        if key in list(reference_dict.keys()):
            keys_to_drop.append(key)

    for key in keys_to_drop:
        target_dict.pop(key, None)
    
    return target_dict

def switch_keys_values(dictionary):
    switched_dict = {}
    
    for key, values in dictionary.items():
        for value in values:
            if value not in switched_dict:
                switched_dict[value] = [key]
            else:
                switched_dict[value].append(key)
    
    return switched_dict

# Tranform Column Data
def transform_cols(df):
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    df = df.astype(str)
    table_obj = {}
    for i in df.columns:
        embeddings = model.encode(" ".join(list(df[i])))
        table_obj[i] = embeddings
    return table_obj

def find_similar(template, table):
    template = transform_cols(template)
    table = transform_cols(table)
    matching_cols = {}
    candidate_matches = {}
    
    for key, value in zip(table.keys(), table.values()):
        
        sim_list = []
        cand_list = []
        for key_temp, value_temp in zip(template.keys(), template.values()):
            sim = util.pytorch_cos_sim(value, value_temp)
            
            if sim >= 0.9:
                sim_list.append(key_temp)
                
            
            elif sim < 0.9 and sim >= 0.5:
                cand_list.append(key_temp)

        matching_cols[key] = sim_list
        candidate_matches[key] = cand_list
            
    matching_cols = {key: value for key, value in matching_cols.items() if value}
    matching_cols = switch_keys_values(matching_cols)
    candidate_matches = {key: value for key, value in candidate_matches.items() if value}
    candidate_matches = switch_keys_values(candidate_matches)
    candidate_matches = drop_keys_and_values(candidate_matches,matching_cols)
                
    return matching_cols, candidate_matches

def analyze_button(template_df, table):
    query = "provide a description of each column in the table"
    prompt = (
        """
            For the following query, if it requires retrieving descriptions of the table reply using the following as a template:
            {
                "column 1" : "description",
                "column 2" : "description",
                "column 3" : "description"
            }
            Ensure that the keys and values are enclosed in double quotes as opposed to single.
            
            Below is the query.
            Query: 
            """
        + query
    )

    response_template = create_and_query_agent(template_df, prompt, query)
    response_table = create_and_query_agent(table, prompt, query)

    return response_template, response_table


                
def session_state_to_dict(session_state, matches, candidates):

    transform_dict = {}
    sess_dict = {}

    print(f"matches:{matches}")
    print(f"candidates:{candidates}")

    for i,j in zip(session_state.row, session_state.row.index):
        sess_dict[j] = i

    print(f"sess dict:{sess_dict}")
    for key, values in matches.items():

        try:

            if len(values) > 1:
                transform_dict[key] = matches[key][0]

            else:
                transform_dict[key] = matches[key][0]
        except Exception:
            continue
    
    for key_2, values_2 in candidates.items():

        try:
            if len(values_2) > 1:

                transform_dict[key_2] =  sess_dict[key_2]
        except Exception:
            continue
    
    return transform_dict

def transformation_query(candidate, template):

    query = "Generate pseudo code in plain english of how to transform the format of the contents of the first dataframe to the format of the contents of the second dataframe. Include details of what specific characters may be removed or added to accomplish the transformation."
    prompt = (
        """
            For the following query, format the response as pesudo code that can then be used as instructions for the GPT completion end-point to follow. If the columns are equal return "No transformations required".
            Include column names in the pseudo code.
            Query: 
            """
        + query
    )
    
    response = create_and_query_agent([candidate,template], prompt, query)
    output = response.__str__()
    
    return output

def get_transformations(transform_dict, candidate_df, template_df):
    
    temp_col = []
    cand_col = []
    response_col = []
    
    for key1, key2 in transform_dict.items():
        template = template_df.filter([key1])
        candidate = candidate_df.filter([key2])
        
        try:
            response = transformation_query(candidate, template)
        except Exception:
            response = '[Error]'
        
        temp_col.append(key1)
        cand_col.append(key2)
        
        response_col.append(response)
    
    df = pd.DataFrame({'Candidate Columns': cand_col, 'Template Columns': temp_col, 'Transformation Description':response_col})
    
    return df

def apply_transformations_agent(query, candidate, template):
    

    query = "Apply the stated transformation to the contents of the first dataframe. The transformations should be applied to the format of the contents. The response should be a python list and be the same length as the original dataframe."
    prompt = (
        """
            For the following query, the response should be formatted as:
            ["transformed contents 1", "transformed contents 2", "transformed contents 3"]
            """
        + query
    )
    try:
        response = create_and_query_agent([candidate,template], prompt, query)
        output = response.__str__()
    except Exception:
        output = "[Error]"
        
    return output

def apply_transformations(transform_df, candidate_df, template_df):
    
    transformed_contents = []
    cols = [] 
    
    for i,j,k in zip(transform_df['Candidate Columns'], transform_df['Template Columns'], transform_df['Transformation Description']):
        if pd.isna(k):
            cols.append(i)
            transformed_contents.append(list(candidate_df[i]))
        else:
            print(f"Instruction: {k}")
            template = template_df.filter([j])
            candidate = candidate_df.filter([i])
            response = apply_transformations_agent(k, candidate, template)
            transformed_contents.append(ast.literal_eval(response))
            cols.append(i)
    
    # Create a dictionary with column names as keys and corresponding data as values
    data_dict = {col_name: col_data for col_name, col_data in zip(cols, transformed_contents)}

    # Create a DataFrame
    df = pd.DataFrame(data_dict)

    return df

def filter_out_matches(matches, df_transform):

    cols_needed = []
    for key,values in matches.items():
        cols_needed.append(values[0])
    
    df = df_transform.filter(cols_needed)

    return df

def is_dict_empty(dictionary):
    if isinstance(dictionary, dict):
        return not bool(dictionary)  # Returns True if the dictionary is empty, otherwise False
    else:
        raise ValueError("Input is not a dictionary")

    
