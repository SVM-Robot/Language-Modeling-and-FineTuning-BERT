from functions import *
from utils import *
from model import *

if __name__ == "__main__":

    print(device)

    print('# 2.1 Weight Tying.')
    hid_size = 300
    emb_size = 300        # <-- changed so that hid_size = emb_size to be able to do Weight Tying
    lr = 0.001                                  
    clip = 3
    vocab_len = len(lang.word2id)

    model = LM_LSTM_2(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    model.apply(init_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    train2(train_loader, dev_loader, test_loader, optimizer, criterion_train, criterion_eval, model, clip)

    print('# 2.2 Variational Dropout.')
    model = LM_LSTM_3(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    model.apply(init_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    train2(train_loader, dev_loader, test_loader, optimizer, criterion_train, criterion_eval, model, clip)

    print('# 2.3 Non-monotonically Triggered AvSGD.')
    model = LM_LSTM_3(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    model.apply(init_weights)
    lr2 = 1.5
    # Two optimizers are defined, they will switch inside the train3 function.
    optimizer = optim.SGD(model.parameters(), lr=lr2)
    optimizer2 = torch.optim.ASGD(model.parameters(), lr=lr2, t0=0, lambd=0., weight_decay=1.2e-6)
    # A switch is present inside the train3 function, with conditions.
    train3(train_loader, dev_loader, test_loader, optimizer, optimizer2, criterion_train, criterion_eval, model, clip)