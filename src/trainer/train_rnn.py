import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torchvision

import random

import torchvision
#from torchvision.transforms import v2

from utils.utils import accuracy_rnn, validate_rnn, img_to_patch


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        return [self.transforms(x), self.transforms(x)]


    
def train(model, train_loader, criterion, optimizer, num_heads, device, supcon_loss=None, eps=0):
    model.train()
    running_loss = 0 
        
    for X, y_true in train_loader:

        optimizer.zero_grad()
        
        if supcon_loss == None:
            X = X.to(device)
            y_true = y_true.to(device)
        else:
            X1 = X[0]
            X2 = X[1]
            X = torch.cat([X1, X2], dim=0).to(device)
            y_true = y_true.to(device)
                        
        # Forward pass
        # -----------------   Multiple Heads loss -----------------
        
        if num_heads != -1:
            train_heads = list(random.sample(range(0, model.seq_len), num_heads))
            
            loss = torch.zeros(num_heads, dtype=torch.float64)
            if supcon_loss == None:                          
                y_hat, _ = model(X, prediction_index=train_heads) 
                for i in range(len(train_heads)):
                    if model.dataset_name == "mixedwm38":
                        loss[i] = criterion(y_hat[i], y_true).sum(dim=1).mean() 
                    else:
                        loss[i] = criterion(y_hat[i], y_true)

                loss = loss.sum() / num_heads
            else:
                shift = 0 
                c = np.zeros(model.seq_len) + 0.1
                y_hat, _, projected = model(X, prediction_index=train_heads) 
                
                projected = torch.stack(projected, dim=0)
                
                bsz = projected.size(1) // 2
                h1, h2 = torch.split(projected, [bsz, bsz], dim=1)
                h = torch.cat([h1.unsqueeze(2), h2.unsqueeze(2)], dim=2) 
                
                y_true_cat = torch.cat([y_true, y_true], dim=0)
                 
                for i in range(num_heads):
                    if model.dataset_name == "mixedwm38":
                        loss[i] = criterion(y_hat[i], y_true_cat).sum(dim=1).mean() + c[train_heads[i]] * supcon_loss(features=h[i], labels=y_true)
                    else:
                        loss[i] = criterion(y_hat[i], y_true_cat) + c[train_heads[i]] * supcon_loss(features=h[i], labels=y_true)
                    
                #print(a)
                
                loss = loss.sum() / num_heads
                
        else:
            y_hat, _ = model(X, prediction_index=-1)
            loss = criterion(y_hat, y_true).sum(dim=1).mean() 
            
        
        # -----------------   -------------------- -----------------
        
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()        
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def training_loop_rnn(model, criterion, optimizer, scheduler, train_loader, valid_loader, 
                      epochs, num_heads, device, supcon_loss=None, eps=0, file_name='model.pth', print_every=1):
 
    best_loss = 1e10
    best_acc = 0
    train_losses = []
    valid_losses = []
    
    if supcon_loss != None:
        train_loader.dataset.set_transform(TwoCropTransform(train_loader.dataset.transforms))
    
    for epoch in range(0, epochs):
        # training
        model, optimizer, train_loss = train(model, train_loader, criterion, optimizer, num_heads, device, supcon_loss=supcon_loss, eps=eps)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            #odel, valid_loss = validate_rnn(model, valid_loader, criterion, device, multioutput=True)
            #alid_losses.append(valid_loss)
            
            if scheduler != None:
                scheduler.step()
       
        #train_acc = accuracy_rnn(model, train_loader, device=device)
        valid_acc = accuracy_rnn(model, valid_loader, device=device, supcon=(supcon_loss != None))
        
        if (valid_acc > best_acc):
            torch.save(model.state_dict(), file_name)
            best_acc = valid_acc
        
        '''
        if (valid_loss < best_loss):
            torch.save(model.state_dict(), file_name)
            best_loss = valid_loss
        '''
        
        if epoch % print_every == (print_every - 1):
                            
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  #'Valid loss: {valid_loss:.4f}\t'
                  #f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    
    return model, (train_losses, valid_losses)