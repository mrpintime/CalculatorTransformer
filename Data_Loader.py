# here we try to create dataset
from torch.utils.data import  DataLoader, Dataset
import operator
import numpy as np
import torch
import torch.nn as nn
import Configs


operats = {'+':operator.add, '-':operator.sub, '/':operator.truediv, '*':operator.mul}
data = {'expression':[], 'target':[]}

for i in range(600):
    num1 = np.random.randint(0,100,(1,))
    num2 = np.random.randint(0,100,(1,))
    op = np.random.choice(['+','-','/', '*'])
    if op == '/':
        if num2.item() == 0:
            continue
        elif num1.item() % num2.item() != 0 :
            num1 = np.random.randint(0,100,(1,)) * num2
    ex = f'{num1[0]}{op}{num2[0]}'
    tg = str(int(operats[op](num1, num2)))
    if ex in data['expression']:
        continue
    data['expression'].append(ex)
    data['target'].append(tg)



# writing the tokenizer

# step1 :  creating vocabulary

vocab_t2i = { # tokenizer
    "[PAD]":1,
    "[SOS]":2,
    "[EOS]":3,
    "0": 4,
    "1": 5,
    "2": 6,
    "3": 7,
    "4": 8,
    "5": 9,
    "6": 10,
    "7": 11,
    "8": 12,
    "9": 13,
    "+": 14,
    "-": 15,
    "/": 16,
    "*": 0
}

vocab_i2t = {i:t for t,i in vocab_t2i.items()}

# step2 : turn expression into list of numbers

def extract_num(strings_list):
    tokenize_list = []
    for t in strings_list:
        for s in t:
            tokenize_list.append(vocab_t2i[s])
    return tokenize_list

exp_tokenized = list(map(extract_num, data['expression']))
tgt_tokenized = list(map(extract_num, data['target']))


# now we should add [sos] [eos] and [pad] to tokenize data
def add_tokens(x):
    x.insert(0,vocab_t2i['[SOS]'])
    x.insert(len(x),vocab_t2i['[EOS]'])
    return x

max_length_src = 15
max_length_tgt = 10

exp_tokenized = list(map(add_tokens, exp_tokenized))
exp_tokenized =  list(map(lambda x: x if len(x) == max_length_src else x + (max_length_src-len(x)) * [vocab_t2i['[PAD]']], exp_tokenized))
tgt_tokenized = list(map(add_tokens, tgt_tokenized))
tgt_tokenized =  list(map(lambda x: x if len(x) == max_length_tgt else x + (max_length_tgt-len(x)) * [vocab_t2i['[PAD]']], tgt_tokenized))


labels, decoder_data =  list(map(lambda x: x[1:] , tgt_tokenized)), list(map(lambda x: x[0:-1] , tgt_tokenized))

encoder_data = exp_tokenized.copy()



class MathDataset(Dataset):

    def __init__(self, ds_raw, config):
        super().__init__()
        self.config = config
        self.ds_raw = ds_raw
        self.sos_token = torch.tensor([vocab_t2i["[SOS]"]], dtype=torch.int64)
        self.eos_token = torch.tensor([vocab_t2i["[EOS]"]], dtype=torch.int64)
        self.pad_token = torch.tensor([vocab_t2i["[PAD]"]], dtype=torch.int64)

    def __len__(self):
        return len(self.ds_raw)

    def __getitem__(self, idx):
        src_text = data['expression']
        tgt_text = data['target']

        encoder_input = torch.tensor(self.ds_raw[0][idx])
        decoder_input = torch.tensor(self.ds_raw[1][idx])
        label = torch.tensor(self.ds_raw[2][idx])
        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.config['seq_len_src']
        assert decoder_input.size(0) == self.config['seq_len_tgt']
        assert label.size(0) == self.config['seq_len_tgt']

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text[idx],
            "tgt_text": tgt_text[idx],
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0




def get_ds(config):
    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(encoder_data))
    
    train_ds_raw = (encoder_data[:train_ds_size], decoder_data[:train_ds_size], labels[:train_ds_size])
    val_ds_raw = (encoder_data[train_ds_size:], decoder_data[train_ds_size:], labels[train_ds_size:])
    
    
    train_ds = MathDataset(train_ds_raw, config)
    val_ds = MathDataset(val_ds_raw, config)


    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, vocab_t2i