import math
import torch
import torch.utils.data as data
from functools import partial
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad()
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    with torch.no_grad():
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

def train2(train_loader, dev_loader, test_loader, optimizer, criterion_train, criterion_eval, model, clip = 5):

    n_epochs = 100
    patience = 7
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    loss = 0
    best_val_loss = []
    stored_loss = 100000000

    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)

        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())

            if loss_dev < stored_loss:
                stored_loss = loss_dev

            pbar.set_description("PPL: %f" % ppl_dev)
            if  ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 4
            else:
                patience -= 1
            if patience <= 0:
                break

            best_val_loss.append(loss_dev)

    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)


# Function to perform Variational Dropout
# Unlike standard dropout where the mask varies for each element in the batch, 
# variational dropout applies the same dropout mask across a specific dimension.

class VariationalDropout(nn.Module):
    def __init__(self, p=0.1):
        super(VariationalDropout, self).__init__()
        self.dropout = p

    def forward(self, x):
        if self.training:
            # x.size(0) = batch size, x.size(1) = sequence length (number of words), x.size(2) = number of embeddings.
            mask = torch.empty(x.size(0), 1, x.size(2), device=device).bernoulli_(1 - self.dropout)
            # Expands the mask to match the shape of x.
            mask = mask.expand(x.size(0), x.size(1), x.size(2))
            # the mask need to be scaled. This scaling is necessary to maintain the expected value of the output,
            # accounting for the dropped elements.
            mask = mask / (1 - self.dropout)                # <-- scaling necessary
            return x * mask
        return x


# train3 function for non-monotonically triggered AvSGD.
# It is similar to train2 but it has an additional optimizer and a switch to ASGD optimizer.
def train3(train_loader, dev_loader, test_loader, optimizer, optimizer2, criterion_train, criterion_eval, model, clip = 5):
    
    n_epochs = 100
    patience = 7
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    loss = 0
    best_val_loss = []
    stored_loss = 100000000
    opt = 'SGD'


    for epoch in pbar:
        
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)

        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())

            if loss_dev < stored_loss:
                stored_loss = loss_dev

            pbar.set_description("PPL: %f" % ppl_dev)
            if  ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 4
            else:
                patience -= 1
            if patience <= 0: 
                break

            best_val_loss.append(loss_dev)

            # <-- Switch to ASGD
            # Take the last 5 values of best_val_loss and check if the loss is increasing.
            # Computes the minimum value of best_val_loss excluding the last 5 elements (best_val_loss[:-5]), 
            # and checks if loss_dev (presumably some measure of loss deviation or development) is greater than this minimum value.
            
            if opt != 'ASGD' and len(best_val_loss)>5 and loss_dev > min(best_val_loss[:-5]):
                print('Switching to ASGD')
                optimizer = optimizer2
                opt = 'ASGD'

    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)