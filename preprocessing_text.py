import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import torch
import pandas as pd
import numpy as np
import csv
import nltk
import pickle as p
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForPreTraining, BertConfig
import logging
import spacy
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert import BertForSequenceClassification, BertForPreTraining
from torch.nn import CrossEntropyLoss, NLLLoss, BCEWithLogitsLoss
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

import argparse

parser = argparse()



def clean(x):
    try:
        return BeautifulSoup(x, features='html.parser').text.replace("\r"," ").replace("\n",".").replace('..','.')
    except TypeError:
        return ''


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):

        features_intermediary = []
        tokens_a = tokenizer.tokenize(example)
        tokens_b = None
        liste_tokens = []
        
        while len(tokens_a) > max_seq_length - 2:
            tokens_b = tokens_a[:(max_seq_length - 2)]
            tokens_a = tokens_a[(max_seq_length - 2):]
            liste_tokens.append(tokens_b)
        liste_tokens.append(tokens_a)
        for tokens_a in liste_tokens:
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)
            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            if output_mode == "classification":
                label_id = label_list[ex_index]
            elif output_mode == "regression":
                label_id = label_list[ex_index]
            else:
                raise KeyError(output_mode)


            features_intermediary.append(
                InputFeatures(input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                label_id=label_id))
        features.append(features_intermediary)
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbose", help="increase output verbosity",
                    action="store_true")
    parser.add_argument("-i","--input_file", type=str,default='kaggle_jobs_dataset.p' help="input pickled dataframe")
    parser.add_argument("-o","--output_file",tyep=str, default='job_dataset.p',help="Name of the output pickled dataframe")
    args = parser.parse_args()
    dataset = p.load(open(args.input_file,'rb'))
    nlp = spacy.load('en', disable=['tokenizer', 'tagger', 'ner', 'textcat'])

### Tokenization of the text
    text_train = list(dataset.Description)
    y_train = dataset.values
    if args.verbose:
        print("Nombre de textes ",len(text_train))
    output_mode = "classification"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_features = convert_examples_to_features(text_train, y_train,512,tokenizer,output_mode)

### Load the model    
    model = BertForPreTraining.from_pretrained('bert-base-uncased')
### Compute the vectors representation of text
    representation_bert = []
    model.eval()
    for text in tqdm(train_features, position=0, leave=True):
        intermediary = []
        for subtext in text:
            intermediary.append(subtext.input_ids)
        with torch.no_grad():        
            subtext = torch.tensor(np.array(intermediary).reshape(-1,512), dtype=torch.long)
            result = model.bert(subtext)[0][11].mean(0).data.numpy()
            torch.cuda.empty_cache()       
        representation_bert.append(np.array(result.mean(0)).reshape(1,-1))
dataset['description_rep'] = representation_bert
p.dump(dataset, open(args.output_file, 'wb'))