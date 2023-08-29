
# %%
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from agent import create_and_query_agent


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

    response_template = create_and_query_agent(template_df, prompt, query, r"C:\Users\Justi\Downloads\output_response1.json")
    response_table = create_and_query_agent(table, prompt, query,r"C:\Users\Justi\Downloads\output_response2.json")

    return response_template, response_table
                