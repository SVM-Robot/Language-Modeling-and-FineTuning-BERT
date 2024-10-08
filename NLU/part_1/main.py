from utils import *
from model import *
from functions import *


if __name__ == "__main__":

    print(device)

    print('#0.0')
    hid_size = 200
    emb_size = 300
    lr = 0.0001
    clip = 5
    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    model = ModelIAS_0(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()
    train2(train_loader, dev_loader, test_loader, optimizer, criterion_slots, criterion_intents, model, lang, clip)

    # ModelIAS_1 is a modified version of ModelIAS_0 with bidirectionality (and hid_size*2).
    print('#1.1 Adding bidirectionality.')
    lr = 0.0001
    model = ModelIAS_1(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train2(train_loader, dev_loader, test_loader, optimizer, criterion_slots, criterion_intents, model, lang, clip)

    # ModelIAS_2 /b/c are a modified version of ModelIAS_1 with dropout layers in different places.
    print('# 1.2 Adding dropout layer: on embedded utterances.')
    lr = 0.0001
    model = ModelIAS_2(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train2(train_loader, dev_loader, test_loader, optimizer, criterion_slots, criterion_intents, model, lang, clip)

    print('# 1.2b Adding dropout layers: on embedded utterances and on LSTM.')
    lr = 0.0001
    model = ModelIAS_2b(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train2(train_loader, dev_loader, test_loader, optimizer, criterion_slots, criterion_intents, model, lang, clip)

    print('# 1.2c Adding dropout layers: on LSTM and on output.')
    lr = 0.0001
    model = ModelIAS_2c(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train2(train_loader, dev_loader, test_loader, optimizer, criterion_slots, criterion_intents, model, lang, clip)
