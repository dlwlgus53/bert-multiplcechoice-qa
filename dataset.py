from datasets import load_dataset
import torch
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer
from dataclasses import dataclass
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
import torch
import pdb



print("Load Tokenizer")
# here, squad means squad2
 # TODO


# {
#     "answer": "A",
#     "article": "\"Schoolgirls have been wearing such short skirts at Paget High School in Branston that they've been ordered to wear trousers ins...",
#     "example_id": "high132.txt",
#     "options": ["short skirts give people the impression of sexualisation", "short skirts are too expensive for parents to afford", "the headmaster doesn't like girls wearing short skirts", "the girls wearing short skirts will be at the risk of being laughed at"],
#     "question": "The girls at Paget High School are not allowed to wear skirts in that    _  ."
# }


            
            
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

class Dataset(torch.utils.data.Dataset):
    def __init__(self,data_name, data_option, tokenizer, max_length, max_options, type):
        self.tokenizer = tokenizer
        self.data_name = data_name
        self.max_length = max_length
        

        try:
            'a'-1
            print("Load processed data")
            with open(f'data/preprocessed_{type}_{data_name}_{data_option}.pickle', 'rb') as f:
                encodings = pickle.load(f)
        except:
            print("preprocess data")
            raw_dataset = load_dataset(self.data_name,data_option)
            input_ids, attention_mask, token_type_ids, labels = self._preprocessing_dataset(raw_dataset[type])
            tokenized_examples = {
                'input_ids' : input_ids,
                'attention_mask' : attention_mask,
                'token_type_ids' : token_type_ids
            }

            print("Encoding dataset (it will takes some time)")
            encodings = {k: [v[i:i+max_options] for i in range(0, len(v), max_options)] for k, v in tokenized_examples.items()}
            encodings['labels'] = labels
            
            
        
            ## save preprocesse data
            with open(f'data/preprocessed_{type}_{data_name}_{data_option}.pickle', 'wb') as f:
                pickle.dump(encodings, f, pickle.HIGHEST_PROTOCOL)

        self.encodings = encodings





    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

    def _preprocessing_dataset(self, dataset):
        input_idss =[]
        input_masks = []
        segment_idss = []

        labels = []
        print(f"preprocessing {self.data_name} data")
        for i, (c, q, os, a) in tqdm(enumerate(zip(dataset['article'], dataset['question'],\
                                    dataset['options'], dataset['answer'])), total= len(dataset['article'])):
            for o in os:
                c_token = tokenizer.tokenize(c)
                q_token = tokenizer.tokenize(q)
                o_token = tokenizer.tokenize(o)
                c_token = self._truncate_c_token(c_token, q_token, o_token)
                
                tokens = ["[CLS]"] + c_token + ["[SEP]"] + q_token + ["[SEP]"] + o_token + ["[SEP]"] 
                segment_ids = [0] * (len(c_token) + 2) + [1] * (len(q_token) + 1)+ [1] * (len(o_token) + 1)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding = [0] * (self.max_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding
                assert self.max_length == len(input_ids) == len(input_mask) == len(segment_ids)
                input_idss.append(input_ids)
                input_masks.append(input_mask)
                segment_idss.append(segment_ids)
                
            label = ord(a) - ord('A')
            labels.append(label)
        
        return input_idss, input_masks, segment_idss, labels
        
    
    def _truncate_c_token(self, c,q,o):
        if len(c) + len(q) + len(o) + 4 > self.max_length:
            c = c[:self.max_length - (len(q) + len(o) + 4)]
        assert(len(c) + len(q) + len(o) + 4 <= self.max_length)
        return c
            
        



if __name__ == '__main__':
    max_length = 128
    max_option = 4
    train_dataset = Dataset('race', 'middle', tokenizer, max_length, max_option, "validation")
    train_loader = DataLoader(train_dataset, 4, shuffle=True)
    for batch in train_loader:
        pdb.set_trace()