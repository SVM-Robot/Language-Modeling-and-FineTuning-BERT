from functions import *
from utils import *
from model import *


if __name__ == "__main__":

    print(device)

    print("# 0.0 base: RNN, SGD, no dropouts.")
    hid_size = 300
    emb_size = 300
    lr = 1.5                                 
    clip = 3
    vocab_len = len(lang.word2id)
    model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    model.apply(init_weights)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum') 
    train2(train_loader, dev_loader, test_loader, optimizer, criterion_train, criterion_eval, model, clip)

    # Replaced LM_RNN with LM_LSTM_0.
    print('# 1.1 Replace RNN with a Long-Short Term Memory (LSTM) network.')
    model = LM_LSTM_0(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    model.apply(init_weights)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train2(train_loader, dev_loader, test_loader, optimizer, criterion_train, criterion_eval, model, clip)

    # LM_LSTM_1 is a modified version of LM_LSTM_0 with two dropout layers.
    print('# 1.2 Add two dropout layers:one after the embedding layer, one before the last linear layer.')
    model = LM_LSTM_1(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    model.apply(init_weights)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train2(train_loader, dev_loader, test_loader, optimizer, criterion_train, criterion_eval, model, clip)

    # Replaced SGD with AdamW.
    print('# 1.3 Replace SGD with AdamW.')
    lr = 0.001                                  
    model = LM_LSTM_1(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    model.apply(init_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr)        # <-- replaced SGD with AdamW
    train2(train_loader, dev_loader, test_loader, optimizer, criterion_train, criterion_eval, model, clip)
   