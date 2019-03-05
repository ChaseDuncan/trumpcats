
from torchtext import data
from torchtext.datasets import LanguageModelingDataset
from torchtext.data import BPTTIterator

import spacy

path_to_data ="/home/cddunca2/data/trump-tweets/tweets.txt"

my_tok = spacy.load('en')
batch_size = 64 
bptt_len = 35
def spacy_tok(x):
    return [tok.text for tok in my_tok.tokenizer(x)]
     
TEXT = data.Field(lower=True, tokenize=spacy_tok)
dataset = LanguageModelingDataset(path_to_data, TEXT)
TEXT.build_vocab(dataset, vectors="glove.6B.200d")

bptt_it = BPTTIterator(dataset, batch_size, bptt_len)

