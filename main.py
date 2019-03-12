import time

from tqdm import tqdm
from torchtext import data
from torchtext.datasets import LanguageModelingDataset
from torchtext.data import BPTTIterator, BucketIterator

import spacy
import torch
import torch.nn as nn

from model import languagemodel
from torchtext.datasets import WikiText2

path_to_data ="/home/chase/data/trump-tweets/tweets.txt"
#path_to_data ="/home/chase/data/trump-tweets/test.txt"
epochs = 50
batch_size = 32
bptt_len = 8
learning_rate = 1e-4

# TODO: this is broken on koyejo-2. possible the wrong pytorch is installed
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

my_tok = spacy.load('en')

def spacy_tok(x):
    return [tok.text for tok in my_tok.tokenizer(x)]

def evaluate():
    model.eval()
    for test_txt in test_it:
        text = test_txt.to(device)
        print(text)
        1/0
    
def train():
    ''' Much of this was copied directly from 
    https://github.com/pytorch/examples/tree/master/word_language_model
    '''
    model.train()
    for batch in tqdm(train_it):
        optimizer.zero_grad()
        text, targets = batch.text.to(device), batch.target.to(device)

        #model.module.init_hidden(batch_size, device)
        model.init_hidden(batch_size, device)
        output, hidden= model(text)

        # pytorch currently only supports cross entropy loss for inputs of 2 or 4 dimensions.
        # we therefore flatten the predictions out across the batch axis so that it becomes
        # shape (batch_size * sequence_length, n_tokens)
        # in accordance to this, we reshape the targets to be
        # shape (batch_size * sequence_length)
        loss = criterion(output.view(-1, ntokens), targets.view(-1))
        loss.backward()
        optimizer.step()

        #torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)


TEXT = data.Field(lower=True, tokenize=spacy_tok)
dataset = LanguageModelingDataset(path_to_data, TEXT)
TEXT.build_vocab(dataset, vectors="glove.6B.200d")

#train, test = dataset.split()
train_it = BPTTIterator(dataset, batch_size, bptt_len)
#train_it = BPTTIterator(train, batch_size, bptt_len)
#test_it = BucketIterator(test)

ntokens = len(TEXT.vocab)
model = languagemodel.LanguageModel(200, 400, ntokens, 40, 0.1)

# Instantiate word embeddings.
weight_matrix = TEXT.vocab.vectors
model.encoder.weight.data.copy_(weight_matrix)

criterion = nn.CrossEntropyLoss()

model = model.to(device)
#model = nn.DataParallel(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# move model to device
for epoch in range(epochs):
    #evaluate()
    print("Training epoch: {}".format(epoch))
    train()
    # store model
    print("Storing checkpoint.")
    torch.save({'epoch': epoch, 
                'specs': '400_hidden__40_layers__0.1_dropout',
                #'model_state_dict': model.module.state_dict(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, 'checkpoints/v4/model.pt')
            
