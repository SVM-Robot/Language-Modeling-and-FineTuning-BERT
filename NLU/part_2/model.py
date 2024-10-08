import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel


# 2. Fine-tuning BERT
# BertModel is used as a base model.
class BERT_1(nn.Module):
    def __init__(self, out_slot, out_int):
        super(BERT_1, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, out_slot)        # <-- slot classifier, 130 classes
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, out_int)       # <-- intent classifier, 26 classes

    def forward(self, input_ids, attention_mask):
        # outputs is a tuple: (last_hidden_state, pooler_output)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        slot_logits = self.slot_classifier(sequence_output)
        intent_logits = self.intent_classifier(pooled_output)
        
        return slot_logits, intent_logits
    

# Same as BERT_1 but without dropout
class BERT_11(nn.Module):
    def __init__(self, out_slot, out_int):
        super(BERT_11, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, out_slot) 
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, out_int)
    
    def forward(self, input_ids, attention_mask):
        # outputs is a tuple: (last_hidden_state, pooler_output)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        slot_logits = self.slot_classifier(sequence_output)
        intent_logits = self.intent_classifier(pooled_output)

        return slot_logits, intent_logits
    

