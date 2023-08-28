
# %%
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# %%
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
    candidate_matches = {key: value for key, value in candidate_matches.items() if value}
                
    return switch_keys_values(matching_cols), switch_keys_values(candidate_matches)
                