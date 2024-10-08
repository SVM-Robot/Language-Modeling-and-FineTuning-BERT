from utils import *
from model import *
from functions import *


if __name__ == "__main__":

    print(device)

    print('# BERT_1')
    lr = 0.0001            
    clip = 5                
    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    model = BERT_1(out_slot, out_int).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()
    train2_bert(train_loader, dev_loader, test_loader, optimizer, criterion_slots, criterion_intents, model, lang, clip, plot_loss=True)

