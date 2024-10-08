from utils import *

# 2.1 Weight Tying
class LM_LSTM_2(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_2, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.drop1 = nn.Dropout(p=emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        self.drop2 = nn.Dropout(p=out_dropout)
        self.output = nn.Linear(hidden_size, output_size)
        self.output.weight = self.embedding.weight              # <-- added for Weight Tying

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        drop1 = self.drop1(emb)
        lstm_out, _  = self.lstm(drop1)
        drop2 = self.drop2(lstm_out)
        output = self.output(drop2).permute(0,2,1)
        return output
    

# 2.2 Variational Dropout
class LM_LSTM_3(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.3,
                 emb_dropout=0.3, n_layers=1):
        super(LM_LSTM_3, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.drop1 = VariationalDropout(p = emb_dropout)        # <-- Variational Dropout layers substituting normal dropout
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        self.drop2 = VariationalDropout(p = out_dropout)       # <-- Variational Dropout layers substituting normal dropout        
        self.output = nn.Linear(hidden_size, output_size)
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        drop1 = self.drop1(emb)
        lstm_out, _  = self.lstm(drop1)
        drop2 = self.drop2(lstm_out)
        output = self.output(drop2).permute(0,2,1)
        return output