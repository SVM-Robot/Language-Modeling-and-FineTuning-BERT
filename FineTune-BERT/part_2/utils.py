import json
from pprint import pprint
import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from transformers import BertTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
PAD_TOKEN = 0

def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

tmp_train_raw = load_data(os.path.join('..','dataset','ATIS','train.json'))
test_raw = load_data(os.path.join('..','dataset','ATIS','test.json'))

portion = 0.10
intents = [x['intent'] for x in tmp_train_raw]
count_y = Counter(intents)
labels = []
inputs = []
mini_train = []

for id_y, y in enumerate(intents):
    if count_y[y] > 1:
        inputs.append(tmp_train_raw[id_y])
        labels.append(y)
    else:
        mini_train.append(tmp_train_raw[id_y])

X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                    random_state=42, 
                                                    shuffle=True,
                                                    stratify=labels)
X_train.extend(mini_train)
train_raw = X_train
dev_raw = X_dev

y_test = [x['intent'] for x in test_raw]


# Lang class modified for BERT:
# Removed word2id as we are using BERT tokenizer.
class Lang():
    def __init__(self, intents, slots, pad_token):
        self.pad_token = pad_token
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = self.pad_token
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

    

words = sum([x['utterance'].split() for x in train_raw], [])
corpus = train_raw + dev_raw + test_raw
slots = set(sum([line['slots'].split() for line in corpus],[]))
intents = set([line['intent'] for line in corpus])

lang = Lang(intents, slots, pad_token=0)



# IntentsAndSlots clas modified for BERT:
class IntentsAndSlotsBert(data.Dataset):
    def __init__(self, dataset, lang, tokenizer, pad_token = 0, unk='unk'):

        self.pad_token = pad_token
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        self.tokenizer = tokenizer

        # Map utterances, slots, and intents to integer IDs
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):

        utt = self.utterances[idx]
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]

        utt_inputs = self.tokenizer(utt, return_tensors="pt", return_token_type_ids=False)
        ids = torch.Tensor(utt_inputs['input_ids'])
        mask = torch.Tensor(utt_inputs['attention_mask'])

        return {
            'ids': ids,
            'mask': mask,
            'slots' : slots,
            'intent': intent,
            'utt': utt
        }


    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    

    def mapping_seq(self, data, mapper): # return: List of tokenized and mapped sequences.

        res = []
        for seq, utt in zip(data, self.utterances):
            tmp_seq = []

            for x, w in zip(seq.split(), utt.split()):
                # we use [1:-1] because we don't want to consider the [CLS] and [SEP] tokens
                tokenized_word = self.tokenizer(w)['input_ids'][1:-1] 
                if x in mapper:
                    # we only map the 1st token and then all the rest is padding
                    tmp_seq.extend([mapper[x]] + [self.pad_token]*(len(tokenized_word)-1)) 
                else:
                    tmp_seq.extend(mapper[self.unk])
                
            # re-add the [CLS] and [SEP] tokens to thesequence    
            res.append([self.pad_token]+tmp_seq+[self.pad_token])
        return res



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_dataset = IntentsAndSlotsBert(train_raw, lang, tokenizer=tokenizer, pad_token=PAD_TOKEN)
dev_dataset = IntentsAndSlotsBert(dev_raw, lang, tokenizer=tokenizer, pad_token=PAD_TOKEN)
test_dataset = IntentsAndSlotsBert(test_raw, lang, tokenizer=tokenizer, pad_token=PAD_TOKEN)


# collate function prepares a batch of variable-length sequences by padding them to the same length, 
# ensuring compatibility with neural network models that require fixed-size input.
# Changed to match (ids, mask, slots, intent) instead of (utterance, slots, intent).
def collate_fn(data):

    pad_token = PAD_TOKEN
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [max(seq.shape) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq 
        padded_seqs = padded_seqs.detach() 
        return padded_seqs, lengths
    
    data.sort(key=lambda x: len(x['ids']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
    

    src_utt, _ = merge(new_item['ids'])
    mask, _ = merge(new_item['mask'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])

    intent = intent.to(device)
    src_utt = src_utt.to(device)
    mask = mask.to(device)
    y_slots = y_slots.to(device)
    
    new_item["ids"] = src_utt
    new_item["mask"] = mask
    new_item["intent"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item

train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=collate_fn)