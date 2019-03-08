import torch
from torchtext import data
from torchtext.datasets import LanguageModelingDataset

import spacy

from model import languagemodel

my_tok = spacy.load('en')
path_to_data ="/home/chase/data/trump-tweets/tweets.txt"

def spacy_tok(x):
    return [tok.text for tok in my_tok.tokenizer(x)]

# Set the random seed manually for reproducibility.
torch.manual_seed(13)
#torch.manual_seed(42)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

TEXT = data.Field(lower=True, tokenize=spacy_tok)
dataset = LanguageModelingDataset(path_to_data, TEXT)
TEXT.build_vocab(dataset, vectors="glove.6B.200d")
ntokens = len(TEXT.vocab)
checkpoint = torch.load("checkpoints/test/model.pt")
model = languagemodel.LanguageModel(200, 200, ntokens, 20, 0.5)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

model.init_hidden(1, device)
model = model.to(device)
input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
temperature = 2.0
numwords = 100
with open("output", 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(numwords):
            output, hidden = model(input)
            word_weights = output.squeeze().div(temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(word_idx)
            word = TEXT.vocab.itos[word_idx]
            if 'https://' in word or 'http://' in word:
                continue
            outf.write(word + ('\n' if i % 20 == 19 else ' '))


            if "eos" in word:
                break
