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
        

        try:
            print("Load processed data")
            with open(f'data/preprocessed_{type}_{data_name}_{data_option}.pickle', 'rb') as f:
                encodings = pickle.load(f)
        except:
            print("preprocess data")
            raw_dataset = load_dataset(self.data_name,data_option)
            contexts, questions_options, labels = self._preprocessing_dataset(raw_dataset[type])

            print("Encoding dataset (it will takes some time)")
            tokenized_examples = tokenizer(contexts, questions_options, max_length = max_length, truncation='only_first', padding=True) # TODO

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
        C = []
        Q_O = []
        labels = []
        print(f"preprocessing {self.data_name} data")
        for i, (c, q, os, a) in tqdm(enumerate(zip(dataset['article'], dataset['question'],dataset['options'], dataset['answer'])), total= len(dataset['article'])):
            for o in os:
                C.append(c)
                Q_O.append(q + ' ' +  o)
            label = ord(a) - ord('A')

            labels.append(label)

        return C, Q_O, labels



if __name__ == '__main__':
    max_length = 128
    max_option = 4
    train_dataset = Dataset('race', 'middle', tokenizer, max_length, max_option, "validation")
    train_loader = DataLoader(train_dataset, 4, shuffle=True)
    for batch in train_loader:
        pdb.set_trace()