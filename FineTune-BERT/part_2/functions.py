import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from utils import *
import torch.nn as nn
from conll import evaluate
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

# train function modified for BERT
def train_loop_bert(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad()
        slots, intent = model(sample['ids'], attention_mask=sample['mask'])
        # to compute loss, slots need to be permuted
        # aftter permuting --> (batch_size, num_labels, sequence_length)
        slots = slots.permute(0,2,1)
        loss_intent = criterion_intents(intent.to(device), sample['intent'])
        loss_slot = criterion_slots(slots.to(device), sample['y_slots'].to(device))
        # Joint training, --> sum of losses
        loss = loss_intent + loss_slot
        loss_array.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step()
    return loss_array

# eval function modified for BERT, see inside comments.
def eval_loop_bert(data, criterion_slots, criterion_intents, model, lang, tokenizer):
    model.eval()
    loss_array = []
    ref_intents = []
    hyp_intents = []
    ref_slots = []
    hyp_slots = []

    with torch.no_grad():

        for sample in data:

            slot_logits, intent_logits = model(sample['ids'], attention_mask=sample['mask'])
            # to compute loss, slots need to be permuted
            slot_logits = slot_logits.permute(0,2,1)
            loss_intent = criterion_intents(intent_logits.to(device), sample['intent'])
            loss_slot = criterion_slots(slot_logits.to(device), sample['y_slots'].to(device))
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())

            out_intents = [lang.id2intent[x] for x in torch.argmax(intent_logits, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intent'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot Inference
            output_slots = torch.argmax(slot_logits, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'][id_seq]
                # we use [1:length-1] instead of [1:length] because the last token is a padding token
                utt_ids = sample['ids'][id_seq][1:length-1].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[1:length-1]]
                utterance = tokenizer.convert_ids_to_tokens(utt_ids)
                to_decode = seq[1:length-1].tolist()
                tmp_seq = []
                tmp_r = []
                for id_el, slot_label in enumerate(gt_slots):
                    
                    # if the slot label is 'pad', it indicates a word piece, 
                    # so the token is concatenated with the previous token to form a 
                    # complete word.
                    if slot_label == 'pad':
                        # we want to reconstruct the word from wordpieces
                        word1 = tmp_r[-1][0]+utterance[id_el]
                        _, label = tmp_r.pop()
                        tmp_r.append((word1, label))
                        continue
                    tmp_r.append((utterance[id_el], slot_label))
                    to_decode_id = to_decode[id_el]
                    tmp_seq.append((utterance[id_el], lang.id2slot[to_decode_id]))

                # example: utterance is ['play', '##ing'] and gt_slots is ['B-activity', 'pad'].
                # On the first iteration, slot_label is 'B-activity', so ('play', 'B-activity') is appended to tmp_r.
                # On the second iteration, slot_label is 'pad', so w is constructed as 'play' + '##ing' = 'playing'.
                # updates tmp_r to ('playing', 'B-activity').

                hyp_slots.append(tmp_seq)
                ref_slots.append(tmp_r)

    try:            
         results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array

# train2_bert wraps train and eval loops.
def train2_bert(train_loader, dev_loader, test_loader, optimizer, criterion_slots, criterion_intents, model, lang, clip, plot_loss=False):

    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0

    for x in tqdm(range(1,n_epochs)):
        loss = train_loop_bert(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=clip)
        if x % 5 == 0:
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop_bert(dev_loader, criterion_slots, criterion_intents, model, lang, tokenizer)
            losses_dev.append(np.asarray(loss_dev).mean())
            
            f1 = results_dev['total']['f']
            if f1 > best_f1:
                best_f1 = f1
                patience = 3
            else:
                patience -= 1
            if patience <= 0:
                break

    results_test, intent_test, _ = eval_loop_bert(test_loader, criterion_slots, criterion_intents, model, lang, tokenizer)    
    print('Slot F1: ', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])


    if (plot_loss):
        plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')
        plt.title('Train and Dev Losses')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.plot(sampled_epochs, losses_train, label='Train loss')
        plt.plot(sampled_epochs, losses_dev, label='Dev loss')
        plt.legend()
        plt.show()


