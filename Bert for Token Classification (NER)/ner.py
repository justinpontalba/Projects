# %% Imports
import os
import pandas as pd
import numpy as np
import json
import re
from nltk.tokenize import sent_tokenize 
from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm
import glob
from sklearn.model_selection import train_test_split
import datetime 
import warnings
warnings.filterwarnings('ignore')

# %% Config
platform = 'Kaggle'
model_name = 'bert-base-uncased'

if platform == 'Kaggle':
    train_path = "C:/Users/Justi/personalGit/Projects/Bert for Token Classification (NER)/train/train/"
    test_path = "C:/Users/Justi/personalGit/Projects/Bert for Token Classification (NER)/test/test/*"

tags_2_idx = {'O': 0 , 'B': 1, 'P': 2}
config = {'MAX_LEN':128,
        'tokenizer': AutoTokenizer.from_pretrained(model_name),
        'batch_size':5,
        'Epoch': 1,
        'train_path':train_path,
        'test_path':test_path, 
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model_name':model_name
        }

# ----------------------------------------------------------------------------------- #
def read_all_json(df, path):
    '''
    This function reads all the json input files and 
    return a dictionary containing the id as the key and all the 
    contents of the json as values
    '''
    text_data = {}
    for i, rec_id in tqdm(enumerate(df.Id), total = len(df.Id)):
        location = f'{path}{rec_id}.json'

        with open(location, 'r') as f:
            text_data[rec_id] = json.load(f)
        
    print("All files read")
    
    return text_data

# Data cleaning and joining functions
def clean_text(txt):
    '''
    This is text cleaning function
    '''
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower())

def data_joining(data_dict_id):
    '''
    This function is to join all the text data from different 
    sections in the json to a single text file. 
    '''
    data_length = len(data_dict_id)

    #     temp = [clean_text(data_dict_id[i]['text']) for i in range(data_length)]
    temp = [data_dict_id[i]['text'] for i in range(data_length)]
    temp = '. '.join(temp)
    
    return temp

def make_shorter_sentence(sentence):
    '''
    This function is to split the long sentences into chunks of shorter sentences upto the 
    maximum length of words specified in config['MAX_LEN']
    '''
    sent_tokenized = sent_tokenize(sentence)
    
    max_length = config['MAX_LEN']
    overlap = 20
    
    final_sentences = []
    
    for tokenized_sent in sent_tokenized:
        sent_tokenized_clean = clean_text(tokenized_sent)
        sent_tokenized_clean = sent_tokenized_clean.replace('.','').rstrip() 
        
        tok_sent = sent_tokenized_clean.split(" ")
        
        if len(tok_sent)<max_length:
            final_sentences.append(sent_tokenized_clean)
        else :
            # print("Making shorter sentences")
            start = 0
            end = len(tok_sent)
            
            for i in range(start, end, max_length-overlap):
                temp = tok_sent[i: (i + max_length)]
                final_sentences.append(" ".join(i for i in temp))

    return final_sentences

def form_labels(sentence, labels_list):
    '''
    This function labels the training data 
    '''
    matched_kwords = []
    matched_token = []
    un_matched_kwords = []
    label = []

    # Since there are many sentences which are more than 512 words,
    # Let's make the max length to be 128 words per sentence.
    tokens = make_shorter_sentence(sentence)
    
    for tok in tokens:    
        tok_split = config['tokenizer'].tokenize(tok)
        
        z = np.array(['O'] * len(tok_split)) # Create final label == len(tokens) of each sentence
        matched_keywords = 0 # Initially no kword matched    

        for kword in labels_list:
            if kword in tok: #This is to first check if the keyword is in the text and then go ahead
                kword_split = config['tokenizer'].tokenize(kword)
                for i in range(len(tok_split)):
                    if tok_split[i: (i + len(kword_split))] == kword_split:
                        matched_keywords += 1

                        if (len(kword_split) == 1):
                            z[i] = 'B'
                        else:
                            z[i] = 'B'
                            z[(i+1) : (i+ len(kword_split))]= 'B'

                        if matched_keywords >1:
                            label[-1] = (z.tolist())
                            matched_token[-1] = tok
                            matched_kwords[-1].append(kword)
                        else:
                            label.append(z.tolist())
                            matched_token.append(tok)
                            matched_kwords.append([kword])
                    else:
                        un_matched_kwords.append(kword)
                
    return matched_token, matched_kwords, label, un_matched_kwords

def labelling(dataset, data_dict):
    '''
    This function is to iterate each of the training data and get it labelled 
    from the form_labels() function.
    '''
    
    Id_list_ = []
    sentences_ = []
    key_ = []
    labels_ = []
    un_mat = []
    un_matched_reviews = 0

    for i, Id in tqdm(enumerate(dataset.Id), total=len(dataset.Id)):

        sentence = data_joining(data_dict[Id])
        labels = train_df.label[train_df.Id == Id].tolist()[0].split("|")
        s, k, l, un_matched = form_labels(sentence=sentence, labels_list = labels)

        if len(s) == 0:
            un_matched_reviews += 1
            un_mat.append(un_matched)
        else: 
            sentences_.append(s)
            key_.append(k)
            labels_.append(l)
            Id_list_.append([Id]*len(l))

    print("Total unmatched keywords:", un_matched_reviews)
    sentences = [item for sublist in sentences_ for item in sublist]
    final_labels = [item for sublist in labels_ for item in sublist]
    keywords = [item for sublist in key_ for item in sublist]
    Id_list = [item for sublist in Id_list_ for item in sublist]
    
    return sentences, final_labels, keywords, Id_list

def dataset_2_list(df):
    id_list = df.id.values.tolist()
    sentences_list = df.train_sentences.values.tolist()
    keywords_list = df.kword.apply(lambda x : eval(x)).values.tolist()
    
    labels_list = df.label.apply(lambda x : eval(x)).values.tolist()    
    labels_list = [list(map(tags_2_idx.get, lab)) for lab in labels_list]
    
    return id_list, sentences_list, keywords_list, labels_list

class form_input():
    def __init__(self, ID, sentence, kword, label, data_type='test'):
        self.id = ID
        self.sentence = sentence
        self.kword = kword
        self.label = label
        self.max_length = config['MAX_LEN']
        self.tokenizer= config['tokenizer']
        self.data_type = data_type
    
    def __len__(self):
        return len(self.sentence)
    
    def __getitem__(self, item):
        toks = config['tokenizer'].tokenize(self.sentence[item])
        label = self.label[item]

        if len(toks) > self.max_length:
            toks = toks[:self.max_length]
            label = label[:self.max_length]

        ########################################
        # Forming the inputs
        ids = config['tokenizer'].convert_tokens_to_ids(toks)
        tok_type_id = [0] * len(ids)
        att_mask = [1] * len(ids)

        # Padding
        pad_len = self.max_length - len(ids)
        ids = ids + [2] * pad_len
        tok_type_id = tok_type_id + [0] * pad_len
        att_mask = att_mask + [0] * pad_len
        ########################################

        # Forming the label
        if self.data_type != 'test':
            label = label + [2] * pad_len
        else:
            label = 1
        
        return {'ids': torch.tensor(ids, dtype = torch.long),
                'tok_type_id': torch.tensor(tok_type_id, dtype = torch.long),
                'att_mask': torch.tensor(att_mask, dtype = torch.long),
                'target': torch.tensor(label, dtype = torch.long)
                }

# Training section
def train_fn(data_loader, model, optimizer):
    '''
    Function to train the model
    '''
    train_loss = 0
    for index, dataset in enumerate(tqdm(data_loader, total = len(data_loader))):
        batch_input_ids = dataset['ids'].to(config['device'], dtype = torch.long)
        batch_att_mask = dataset['att_mask'].to(config['device'], dtype = torch.long)
        batch_tok_type_id = dataset['tok_type_id'].to(config['device'], dtype = torch.long)

        batch_target = dataset['target'].to(config['device'], dtype = torch.long)

        output = model(batch_input_ids,
                        token_type_ids = None,
                        attention_mask = batch_att_mask,
                        labels = batch_target)
        
        step_loss = output[0]
        prediction = output[1]

        step_loss.sum().backward()
        optimizer.step()
        train_loss +=step_loss
        optimizer.zero_grad()

    return train_loss.sum()

def eval_fn(data_loader, model):
    '''
    Function to evaluate the model on each epoch.
    We can also use Jaccard metric to see the performance on each epoch
    '''

    model.eval()

    eval_loss = 0
    predictions = np.array([], dtype = np.int64).reshape(0, config['MAX_LEN'])
    true_labels = np.array([], dtype = np.int64).reshape(0, config['MAX_LEN'])

    with torch.no_grad():
        for index, dataset in enumerate(tqdm(data_loader, total = len(data_loader))):
            batch_input_ids = dataset['ids'].to(config['device'], dtype = torch.long)
            batch_att_mask = dataset['att_mask'].to(config['device'], dtype = torch.long)
            batch_tok_type_id = dataset['tok_type_id'].to(config['device'], dtype = torch.long)

            batch_target = dataset['target'].to(config['device'], dtype = torch.long)

            output = model(batch_input_ids,
                            token_type_ids = None,
                            attention_mask = batch_att_mask,
                            labels = batch_target)
            
            step_loss = output[0]
            eval_prediction = output[1]

            eval_loss += step_loss

            eval_prediction = np.argmax(eval_prediction.detach().to('cpu').numpy(), axis = 2)

            actual = batch_target.to('cpu').numpy()

            predictions = np.concatenate((predictions, eval_prediction), axis = 0)
            true_labels = np.concatenate((true_labels, actual), axis = 0)
        
    return eval_loss.sum(), predictions, true_labels


def train_engine(epoch, train_data, valid_data):
    model = transformers.BertForTokenClassification.from_pretrained('bert-base-uncased',  num_labels = len(tags_2_idx))
    model = nn.DataParallel(model)
    model = model.to(config['device'])

    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr = 3e-5)

    best_eval_loss = 1000000
    for i in range(epoch):
        train_loss = train_fn(data_loader = train_data,
                              model = model,
                              optimizer = optimizer)
        eval_loss, eval_predictions, true_labels = eval_fn(data_loader = valid_data, model = model)

        print(f"Epoch {i}, Train loss: {train_loss}, Eval loss: {eval_loss}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            
            print("Saving the model")
            torch.save(model.state_dict(), f"model_{epoch}_bert_base_uncased.bin")

    return model, eval_predictions, true_labels

def read_test_json(test_data_folder):
    '''
    This function reads all the json input files and return a dictionary containing the id as the key
    and all the contents of the json as values
    '''

    test_text_data = {}
    total_files = len(glob.glob(test_data_folder))
    print(f"total files: {total_files}")
    
    for i, test_json_loc in enumerate(glob.glob(test_data_folder)):
        filename = test_json_loc.split("/")[-1][:-5]

        with open(test_json_loc, 'r') as f:
            test_text_data[filename] = json.load(f)

        if (i%1000) == 0:
            print(f"Completed {i}/{total_files}")

    print("All files read")
    return test_text_data

# Prediction
def prediction_fn(tokenized_sub_sentence):

    tkns = tokenized_sub_sentence
    indexed_tokens = config['tokenizer'].convert_tokens_to_ids(tkns)
    segments_ids = [0] * len(indexed_tokens)

    tokens_tensor = torch.tensor([indexed_tokens]).to(config['device'])
    segments_tensors = torch.tensor([segments_ids]).to(config['device'])
    
    model.eval()
    with torch.no_grad():
        logit = model(tokens_tensor, 
                      token_type_ids=None,
                      attention_mask=segments_tensors)

        logit_new = logit[0].argmax(2).detach().cpu().numpy().tolist()
        prediction = logit_new[0]

        kword = ''
        kword_list = []

        for k, j in enumerate(prediction):
            if (len(prediction)>1):

                if (j!=0) & (k==0):
                    #if it's the first word in the first position
                    #print('At begin first word')
                    begin = tkns[k]
                    kword = begin

                elif (j!=0) & (k>=1) & (prediction[k-1]==0):
                    #begin word is in the middle of the sentence
                    begin = tkns[k]
                    previous = tkns[k-1]

                    if begin.startswith('##'):
                        kword = previous + begin[2:]
                    else:
                        kword = begin

                    if k == (len(prediction) - 1):
                        #print('begin and end word is the last word of the sentence')
                        kword_list.append(kword.rstrip().lstrip())

                elif (j!=0) & (k>=1) & (prediction[k-1]!=0):
                    # intermediate word of the same keyword
                    inter = tkns[k]

                    if inter.startswith('##'):
                        kword = kword + "" + inter[2:]
                    else:
                        kword = kword + " " + inter


                    if k == (len(prediction) - 1):
                        #print('begin and end')
                        kword_list.append(kword.rstrip().lstrip())

                elif (j==0) & (k>=1) & (prediction[k-1] !=0):
                    # End of a keywords but not end of sentence.
                    kword_list.append(kword.rstrip().lstrip())
                    kword = ''
                    inter = ''
            else:
                if (j!=0):
                    begin = tkns[k]
                    kword = begin
                    kword_list.append(kword.rstrip().lstrip())

    return kword_list

def long_sent_split(long_tokens):
    '''
    If the token length is >the max length this function splits it into 
    mutiple list of specified smaller max_length
    '''
    
    start = 0
    end = len(long_tokens)
    max_length = 64

    final_long_tok_split = []
    for i in range(start, end, max_length):
        temp = long_tokens[i: (i + max_length)]
        final_long_tok_split.append(temp)
    return final_long_tok_split

def get_predictions(data_dict):
    
    results = {}

    for i, Id in enumerate(data_dict.keys()):
        current_id_predictions = []
        
#         print(Id)
        sentences = data_joining(data_dict[Id])
        sentence_tokens = sent_tokenize(sentences)
        
        for sub_sentence in sentence_tokens:
            cleaned_sub_sentence = clean_text(sub_sentence)
        
            # Tokenize the sentence
            tokenized_sub_sentence = config['tokenizer'].tokenize(cleaned_sub_sentence)
            
            if len(tokenized_sub_sentence) == 0:
                # If the tokenized sentence are empty
                sub_sentence_prediction_kword_list = []
                
            elif len(tokenized_sub_sentence) <= 512:
                # If the tokenized sentence are less than 512
                sub_sentence_prediction_kword_list = prediction_fn(tokenized_sub_sentence)

            else:
                # If the tokenized sentence are >512 which is long sentences
                long_sent_kword_list = []
                
                tokenized_sub_sentence_tok_split = long_sent_split(tokenized_sub_sentence)
                for i, sent_tok in enumerate(tokenized_sub_sentence_tok_split):
                    if len(sent_tok) != 0:
                        kword_list = prediction_fn(sent_tok)
                        long_sent_kword_list.append(kword_list)
                flat_long_sent_kword = [item for sublist in long_sent_kword_list for item in sublist]
                sub_sentence_prediction_kword_list = flat_long_sent_kword
                            
            if len(sub_sentence_prediction_kword_list) !=0:
                current_id_predictions = current_id_predictions + sub_sentence_prediction_kword_list

        results[Id] = list(set(current_id_predictions))
                
    print("All predictions completed")
    
    return results


# ----------------------------------------------------------------------------------- #

if __name__ == "__main__":
        
    # Read the train data and combine the labels together
    train = pd.read_csv(r"C:\Users\Justi\personalGit\Projects\Bert for Token Classification (NER)\train\train.csv")
    train_df = train.groupby(['Id']).agg(label_count = ('cleaned_label', 'count'), label = ('cleaned_label', '|'.join)).reset_index()
    print(train_df)
    train_df = train_df[0:5000]

    # Reading all the json train files
    train_data_dict = read_all_json(df=train_df, path=config['train_path'])
    train_sentences, train_labels, train_keywords, train_Id_list = labelling(dataset = train_df, data_dict=train_data_dict)

    print("")
    print(f" train sentences: {len(train_sentences)}, train label: {len(train_labels)}, train keywords: {len(train_keywords)}, train_id list: {len(train_Id_list)}")

    unique_df = pd.DataFrame({'id':train_Id_list, 
                            'train_sentences': train_sentences, 
                            'kword': train_keywords, 
                            'label':train_labels})
    unique_df.label = unique_df.label.astype('str')
    unique_df.kword = unique_df.kword.astype('str')
    unique_df['sent_len'] = unique_df.train_sentences.apply(lambda x : len(x.split(" ")))
    unique_df = unique_df.drop_duplicates()
    print(unique_df)
    print(unique_df.shape)

    # % Taking a sample of the dataset
    unique_df = unique_df.sample(int(unique_df.shape[0]*0.10)).reset_index(drop=True)

    # %% Train and validation split
    np.random.seed(100)
    train_df, valid_df = train_test_split(unique_df, test_size = 0.2)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    print(train_df.shape, valid_df.shape)

    # %% Converting the Dataframe back to list
    print("Let's look at a simple example")
    print("Train sentence:", unique_df.train_sentences[0])
    print("Train label:", unique_df.kword[0])
    print("Train label:", unique_df.label[0])

    final_train_id_list, final_train_sentences, final_train_keywords, final_train_labels = dataset_2_list(df=train_df)
    final_valid_id_list, final_valid_sentences, final_valid_keywords, final_valid_labels = dataset_2_list(df=valid_df)

    # %% Forming the input and defining the dataloader
    train_prod_input = form_input(ID = final_train_id_list,
                                  sentence = final_train_sentences,
                                  kword = final_train_keywords,
                                  label = final_train_labels,
                                  data_type='train')
    valid_prod_input = form_input(ID = final_valid_id_list,
                                  sentence = final_valid_sentences,
                                  kword = final_valid_keywords,
                                  label = final_valid_labels,
                                  data_type='valid')
    train_prod_input_data_loader = DataLoader(train_prod_input,
                                              batch_size = config['batch_size'],
                                              shuffle=True)
    valid_prod_input_data_loader = DataLoader(valid_prod_input,
                                              batch_size = config['batch_size'],
                                              shuffle=True)
    
    # Checking a sample input
    ind = 1
    print("Input sentence:")
    print(final_train_sentences[ind])

    print("")
    print("Input label:")
    print(final_train_keywords[ind])

    print("")
    print("Output:")
    train_prod_input[ind]

    # Model initialization and train
    model, val_predictions, val_true_label = train_engine(epoch = config['Epoch'],
                                                                         train_data = train_prod_input_data_loader,
                                                                         valid_data = valid_prod_input_data_loader)

    # Reading the test data    
    test_data_dict = read_test_json(test_data_folder=config['test_path'])

    results = get_predictions(data_dict = test_data_dict)

    sub_df = pd.DataFrame({'Id': list(results.keys()),
                       'PredictionString': list(results.values())})
    sub_df.PredictionString = sub_df.PredictionString.apply(lambda x : "|".join(x))
    print(sub_df)

    sub_df.to_csv("submission.csv", index=False)