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



class Dataset(torch.utils.data.Dataset):
    def __init__(self,data_name, data_option, tokenizer, max_length, max_options, type, logger):
        self.tokenizer = tokenizer
        self.data_name = data_name
        self.max_length = max_length
        self.max_options = max_options
        self.logger = logger
        try:
            logger.info("Load processed data")
            with open(f'data/preprocessed_{type}_{data_name}_{data_option}_{max_length}_{max_options}.pickle', 'rb') as f:
                encodings = pickle.load(f)
        except:
            logger.info("preprocessing data...")
            if data_option:
                raw_dataset = load_dataset(self.data_name,data_option)
            else:
                raw_dataset = load_dataset(self.data_name)
                
                
            input_ids, attention_mask, token_type_ids, labels = self._preprocessing_dataset(raw_dataset[type])
            tokenized_examples = {
                'input_ids' : input_ids,
                'attention_mask' : attention_mask,
                'token_type_ids' : token_type_ids
            }

            logger.info("Encoding dataset (it will takes some time)")
            encodings = {k: [v[i:i+max_options] for i in range(0, len(v), max_options)] for k, v in tokenized_examples.items()}
            encodings['labels'] = labels
            
            assert len(encodings['labels']) == len(encodings['attention_mask']) == len(encodings['input_ids']) == len(encodings['token_type_ids'])
            
        
            ## save preprocesse data
            with open(f'data/preprocessed_{type}_{data_name}_{data_option}_{max_length}_{max_options}.pickle', 'wb') as f:
                pickle.dump(encodings, f, pickle.HIGHEST_PROTOCOL)

        self.encodings = encodings
        # print(self.tokenizer.convert_ids_to_tokens(encodings['input_ids'][0][0]))
        

    def __getitem__(self, idx):
        temp = {}
        try:
            temp = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        except:
            pdb.set_trace()
        return temp

    def __len__(self):
        return len(self.encodings['input_ids'])

    def _preprocessing_dataset(self, dataset):
        input_idss =[]
        input_masks = []
        segment_idss = []

        labels = []
        self.logger.info(f"preprocessing {self.data_name} data")
        if self.data_name == 'race':
            article, question, options, answer = dataset['article'], dataset['question'], dataset['options'], dataset['answer']
        elif self.data_name == 'dream':
            article, question, options, answer = dataset['dialogue'], dataset['question'], dataset['choice'], dataset['answer']
        
        
        for i, (c, q, os, a) in enumerate(zip(article, question, options, answer)):
            os += ['not mentioned']
            os += (['wrong'] * (self.max_options-len(os)))
            assert len(os) == self.max_options
            for o in os:
                if self.data_name == 'dream':
                    c = ' '.join(c)
                c_token = self.tokenizer.tokenize(c)
                q_token = self.tokenizer.tokenize(q)
                o_token = self.tokenizer.tokenize(o)
                c_token, q_token, o_token = self._truncate_cqo_token(c_token, q_token, o_token)
                
                tokens = ["[CLS]"] + c_token + ["[SEP]"] + q_token + o_token + ["[SEP]"] 
                segment_ids = [0] * (len(c_token) + 2) + [1] * (len(q_token) )+ [1] * (len(o_token) + 1)
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
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

            if self.data_name == 'race':
                label = ord(a) - ord('A')
            elif self.data_name == 'dream':
                label = os.index(a)
                
            labels.append(label)
        
        return input_idss, input_masks, segment_idss, labels
        
    
    
    
    def _truncate_cqo_token(self, c,q,o):
        special_token_num = 3 # [cls][sep][sep]
        
        if len(c) + len(q) + len(o) + special_token_num > self.max_length:
            if self.max_length - (len(q) + len(o) + special_token_num) > 0:
                c = c[:self.max_length - (len(q) + len(o) + special_token_num)]
            else:
                c = []
        if len(c) + len(q) + len(o) + special_token_num > self.max_length:
            if self.max_length - (len(o) + special_token_num) > 0:
                q = q[:self.max_length - (len(o) + special_token_num)]
            else:
                q = []
                
        if len(c) + len(q) + len(o) + special_token_num > self.max_length:
            if self.max_length - special_token_num > 0:
                o = o[:self.max_length - ( special_token_num)]
            else:
                o = []
        if (len(c) + len(q) + len(o) + special_token_num > self.max_length):
            pdb.set_trace()
        return c,q,o
            
        



if __name__ == '__main__':
    print("Load Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    max_length = 128
    max_option = 6
    train_dataset = Dataset('dream', None, tokenizer, max_length, max_option, "validation")
    train_loader = DataLoader(train_dataset, 4, shuffle=True)
    for batch in train_loader:
        pdb.set_trace()