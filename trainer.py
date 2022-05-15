
from sia_transformer import Seq2SeqTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from sklearn.utils import shuffle
import json

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import sia_vm
import time
import math
from tqdm import tqdm
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import random

random.seed(42)

def create_mask(src, tgt, PAD_IDX):
    seq_len_src = src.shape[0]
    seq_len_tgt = tgt.shape[0]

    mask_src = torch.zeros((seq_len_src, seq_len_src), device=device).type(torch.bool)
    mask_tgt = generate_square_subsequent_mask(seq_len_tgt)

    padding_mask_src = (src == PAD_IDX).transpose(0, 1)
    padding_mask_tgt = (tgt == PAD_IDX).transpose(0, 1)
    
    return mask_src, mask_tgt, padding_mask_src, padding_mask_tgt


def generate_square_subsequent_mask(seq_len):
    mask = (torch.triu(torch.ones((seq_len, seq_len), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

manager =  sia_vm.SIABinaryManager(False)
vocab_size = manager.total_dict_len() + 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df = pd.read_csv("./output/sia_compiled.csv")

train_data = []

x_maxlen = np.max(df["train_X"].apply(lambda x: len(manager.encode_s(x))))
y_maxlen = np.max(df["train_Y"].apply(lambda x: len(x.replace("  ", " ").split(" "))))

x_maxlen = max(x_maxlen, 100) + 1
y_maxlen = max(y_maxlen, 50)

print("Loading data...")
print("")

PAD_IDX = manager.encode_word(" ")
START_IDX = manager.encode_word("<BEGIN>")
END_IDX = manager.encode_word("<EOS>")

for x, y in zip(df["train_X"].values, df["train_Y"].values):
    ex = manager.encode_s(x)
    ey = [int(x) for x in (y.replace("  ", " ").split(" "))]
    train_X = torch.tensor(ex + [manager.encode_word("<EOS>")] + [PAD_IDX] * (x_maxlen - len(ex)), device=device)
    train_Y = torch.tensor(ey + [PAD_IDX] * (y_maxlen - len(ey)), device=device)
    train_data.append((train_X, train_Y))

random.shuffle(train_data)
train_data, test_data = train_test_split(train_data, train_size=0.9)

batch_size = 2
epoch = 60 # 180

def generate_batch(data_batch):
    batch_src, batch_tgt = [], []
    for src, tgt in data_batch:
        batch_src.append(src)
        batch_tgt.append(tgt)
        
    batch_src = pad_sequence(batch_src, padding_value=PAD_IDX)
    batch_tgt = pad_sequence(batch_tgt, padding_value=PAD_IDX)
    
    return batch_src, batch_tgt

train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)
test_iter  = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)

print("Total training_data size: {}".format(len(train_data)))

def train(model, data, optimizer, criterion, PAD_IDX):
    model.train()
    losses = 0
    for src, tgt in tqdm(data):
        
        src = src.to(device)
        tgt = tgt.to(device)

        input_tgt = tgt[:-1, :]

        mask_src, mask_tgt, padding_mask_src, padding_mask_tgt = create_mask(src, input_tgt, PAD_IDX)
        
        logits = model(
            src=src, tgt=input_tgt,
            mask_src=mask_src, mask_tgt=mask_tgt,
            padding_mask_src=padding_mask_src, padding_mask_tgt=padding_mask_tgt,
            memory_key_padding_mask=padding_mask_src
        )

        optimizer.zero_grad()

        output_tgt = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), output_tgt.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        
    return losses / len(data)

def evaluate(model, data, criterion, PAD_IDX):
    model.eval()
    losses = 0
    for src, tgt in data:
        src = src.to(device)
        tgt = tgt.to(device)

        input_tgt = tgt[:-1, :]

        mask_src, mask_tgt, padding_mask_src, padding_mask_tgt = create_mask(src, input_tgt, PAD_IDX)

        logits = model(
            src=src, tgt=input_tgt,
            mask_src=mask_src, mask_tgt=mask_tgt,
            padding_mask_src=padding_mask_src, padding_mask_tgt=padding_mask_tgt,
            memory_key_padding_mask=padding_mask_src
        )
        
        output_tgt = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), output_tgt.reshape(-1))
        losses += loss.item()
        
    return losses / len(data)


embedding_size = 32 # 64
nhead = 8

dim_feedforward = 100
num_encoder_layers = 2 #6
num_decoder_layers = 2 #6
dropout = 0.1

print("Initialize model...")
print("")

model = Seq2SeqTransformer(
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    embedding_size=embedding_size,
    vocab_size_src=vocab_size, vocab_size_tgt=vocab_size,
    dim_feedforward=dim_feedforward,
    dropout=dropout, nhead=nhead
)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

model = model.to(device)

criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(model.parameters())

best_loss = float('Inf')
best_model = None
patience = 10
counter = 0

print("Start training...")
print("")

for loop in range(1, epoch + 1):
    start_time = time.time()
    loss_train = train(
        model=model, data=train_iter, optimizer=optimizer,
        criterion=criterion, PAD_IDX=PAD_IDX
    )
    elapsed_time = time.time() - start_time
    loss_valid = evaluate(
        model=model, data=test_iter, criterion=criterion, PAD_IDX=PAD_IDX
    )
    
    print('[{}/{}] train loss: {:.2f}, valid loss: {:.2f}  [{}{:.0f}s] counter: {} {}'.format(
        loop, epoch,
        loss_train, loss_valid,
        str(int(math.floor(elapsed_time / 60))) + 'm' if math.floor(elapsed_time / 60) > 0 else '',
        elapsed_time % 60,
        counter,
        '**' if best_loss > loss_valid else ''
    ))
    
    if best_loss > loss_valid:
        best_loss = loss_valid
        best_model = model
        counter = 0
        
    if counter > patience:
        break
    
    counter += 1
    
torch.save(best_model.state_dict(), 'transformer.pth')
#torch.save(best_model.to('cpu').state_dict(), 'transformer.pth')
config = {
    "num_encoder_layers": num_encoder_layers,
    "num_decoder_layers": num_decoder_layers,
    "embedding_size": embedding_size,
    "vocab_size": vocab_size,
    "dim_feedforward": dim_feedforward,
    "dropout": dropout,
    "nhead": nhead,
    "y_maxlen": y_maxlen
}

with open("./t-model-config.json", 'w') as f:
    s = json.dumps(config)
    f.write(s)