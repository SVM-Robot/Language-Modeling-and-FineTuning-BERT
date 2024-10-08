import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 0.0
class ModelIAS_0(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        super(ModelIAS_0, self).__init__()

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=False, batch_first=True)    
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, utterance, seq_lengths):
        utt_emb = self.embedding(utterance)
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        last_hidden = last_hidden[-1,:,:]
        slots = self.slot_out(utt_encoded)
        intent = self.intent_out(last_hidden)
        slots = slots.permute(0,2,1)
        return slots, intent
    
# 1.1 Adding bidirectionality
class ModelIAS_1(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        super(ModelIAS_1, self).__init__()
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True)    # <-- bidirectionality
        self.slot_out = nn.Linear(hid_size * 2, out_slot)         # <-- hid_size * 2 because of bidirectionality
        self.intent_out = nn.Linear(hid_size * 2, out_int)        # <-- hid_size * 2 because of bidirectionality
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, utterance, seq_lengths):

        utt_emb = self.embedding(utterance)
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        last_hidden = torch.cat((last_hidden[-2,:,:], last_hidden[-1,:,:]), dim=1)
       
        slots = self.slot_out(utt_encoded)
        intent = self.intent_out(last_hidden)
        slots = slots.permute(0,2,1)

        return slots, intent
    
# 1.2 Adding dropout layer
class ModelIAS_2(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        super(ModelIAS_2, self).__init__()
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True)    
        self.dropout_emb = nn.Dropout(0.1)              # <-- dropout layer on embedded utterances
        self.slot_out = nn.Linear(hid_size * 2, out_slot) 
        self.intent_out = nn.Linear(hid_size * 2, out_int) 
        
    def forward(self, utterance, seq_lengths):

        utt_emb = self.embedding(utterance)
        utt_emb = self.dropout_emb(utt_emb)                         # Apply dropout to the embedded utterances
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        last_hidden = torch.cat((last_hidden[-2,:,:], last_hidden[-1,:,:]), dim=1)
        slots = self.slot_out(utt_encoded)
        intent = self.intent_out(last_hidden)
        slots = slots.permute(0,2,1)

        return slots, intent
    
# 1.1b Two dropout layers, utt and LSTM
class ModelIAS_2b(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        super(ModelIAS_2b, self).__init__()
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True)    
        self.dropout_emb = nn.Dropout(0.1)
        self.dropout_lstm = nn.Dropout(0.2)                         # Additional dropout after LSTM
        self.slot_out = nn.Linear(hid_size * 2, out_slot) 
        self.intent_out = nn.Linear(hid_size * 2, out_int) 
        
    def forward(self, utterance, seq_lengths):

        utt_emb = self.embedding(utterance)
        utt_emb = self.dropout_emb(utt_emb)                         # dropout layer on embedded utterances     
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        utt_encoded = self.dropout_lstm(utt_encoded)                # dropout layer after LSTM
        last_hidden = torch.cat((last_hidden[-2,:,:], last_hidden[-1,:,:]), dim=1)
        slots = self.slot_out(utt_encoded)
        intent = self.intent_out(last_hidden)
        slots = slots.permute(0,2,1)

        return slots, intent
    

# 1.1c Two dropout layers: LSTM and output
class ModelIAS_2c(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        super(ModelIAS_2c, self).__init__()
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True)                      
        self.slot_out = nn.Linear(hid_size * 2, out_slot) 
        self.intent_out = nn.Linear(hid_size * 2, out_int) 
        self.dropout = nn.Dropout(0.3)

    def forward(self, utterance, seq_lengths):
        utt_emb = self.embedding(utterance)
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        utt_encoded = self.dropout(utt_encoded)                 # dropout layer after LSTM   
        last_hidden = torch.cat((last_hidden[-2,:,:], last_hidden[-1,:,:]), dim=1)
        last_hidden = self.dropout(last_hidden)                 # dropout layer before output
        
        slots = self.slot_out(utt_encoded)
        intent = self.intent_out(last_hidden)
        slots = slots.permute(0,2,1)

        return slots, intent
