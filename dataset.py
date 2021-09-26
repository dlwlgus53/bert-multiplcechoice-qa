from datasets import load_dataset
import torch
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
import pdb



print("Load Tokenizer")
# here, squad means squad2
max_of_options = 10 # TODO 


#  "answer": "A",
#     "article": "\"Schoolgirls have been wearing such short skirts at Paget High School in Branston that they've been ordered to wear trousers ins...",
#     "example_id": "high132.txt",
#     "options": ["short skirts give people the impression of sexualisation", "short skirts are too expensive for parents to afford", "the headmaster doesn't like girls wearing short skirts", "the girls wearing short skirts will be at the risk of being laughed at"],
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

def preprocessing_dataset(examples,data_name, max_of_options, tokenizer):
    if data_name == "race":
        # Repeat each first sentence four times to go with the four possibilities of second sentences.
        first_sentences = [[context] * max_of_options for context in examples["article"]]
        # Grab all second sentences possible for each context.
        second_sentences = examples['options']
        # pad with wrong value
        for sentences in second_sentences:
            sentences += ['wrong'] * (max_of_options - len(sentences))

        # Flatten everything
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])
        
        # Tokenize
        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation='only_first')
        # Un-flatten
        return {k: [v[i:i+max_of_options] for i in range(0, len(v), max_of_options)] for k, v in tokenized_examples.items()}


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features,answer):
        label_name = 'answer'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = 
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        # print(flattened_features)
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

'''
{
    "answer": "B",
    "article": "\"There is not enough oil in the world now. As time goes by, it becomes less and less, so what are we going to do when it runs ou...",
    "example_id": "middle3.txt",
    "options": ["There is more petroleum than we can use now.", "Trees are needed for some other things besides making gas.", "We got electricity from ocean tides in the old days.", "Gas wasn't used to run cars in the Second World War."],
    "question": "According to the passage, which of the following statements is TRUE?"
}
'''

if __name__ == "__main__":
    rawdata = load_dataset('race','middle')
    max_of_options = 5
    features = preprocessing_dataset(rawdata['train'][:4], 'race', max_of_options, tokenizer)

    accepted_keys = ["input_ids", "attention_mask", "answer"]
    features = [{k: v for k, v in encoded_datasets["train"][i].items() if k in accepted_keys} for i in range(10)]
    batch = DataCollatorForMultipleChoice(tokenizer)(features)
