import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import random

from utils.utils import *
#from src.utils.utils import *



class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        return [self.transforms(x), self.transforms(x)]


def train_rnn(
    confidnet, classifier, train_loader, 
    criterion, optimizer, num_heads, device, 
    supcon_classifier=True, supcon_confid_loss=None
):
    confidnet.train()
    freeze_parameters(classifier)
    
    running_loss = 0
    supcon_confid = supcon_confid_loss != None
           
    for X, y_true in train_loader:

        optimizer.zero_grad()
        
        if supcon_confid:
            X1 = X[0]
            X2 = X[1]
            X = torch.cat([X1, X2], dim=0).to(device)
        
        X = X.to(device)
        y_true = y_true.to(device)
        
                
        # Forward pass
        # -----------------   Multiple Heads loss -----------------
        
        with torch.no_grad():
            if supcon_classifier:
                y_hat, hidden, _  = classifier(X, multioutput=True)
            else:
                y_hat, hidden = classifier(X, multioutput=True)
            
        hidden = torch.stack(hidden, dim=0)
        y_hat = torch.stack(y_hat, dim=0)
        
   
        if not supcon_confid:
            confidence, _ = confidnet(hidden.transpose(0, 1), multioutput=True)
        else:
            c = 0.1
            confidence, _, projected = confidnet(hidden.transpose(0, 1), multioutput=True)
            projected = torch.stack(projected, dim=0)
                
            bsz = projected.size(1) // 2
            h1, h2 = torch.split(projected, [bsz, bsz], dim=1)
            h = torch.cat([h1.unsqueeze(2), h2.unsqueeze(2)], dim=2) 
            y_true_cat = torch.cat([y_true, y_true], dim=0)
            #y_hat_cat = torch.cat([y_hat, y_hat], dim=0)
            y_hat_cat = y_hat
            
        confidence = torch.stack(confidence, dim=0)
                
        if num_heads != -1:
            train_heads = list(random.sample(range(0, confidnet.seq_len), num_heads))            
            loss = torch.zeros(num_heads, dtype=torch.float64)
            for i in range(len(train_heads)):                
                if supcon_confid:
                    loss[i] = criterion(confidence[i], y_hat_cat, y_true_cat).mean() + c * supcon_confid_loss(features=h[i], labels=y_true)
                else:
                    loss[i] = criterion(confidence[i], y_hat, y_true).mean()
               
            loss = loss.sum() / num_heads
        else:
            if supcon_confid:
                loss = criterion(confidence, y_hat_cat, y_true_cat).mean()
            else:
                loss = criterion(confidence, y_hat, y_true).mean()
                                       
            if supcon_confid:
                for i in range(confidence.size(0)):  
                    loss += c * supcon_confid_loss(features=h[i], labels=y_true) / confidence.size(0)
              
        # -----------------   -------------------- -----------------
        
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(confidnet.parameters(), max_norm=0.5)
        optimizer.step()   
        
        
    epoch_loss = running_loss / len(train_loader.dataset)
    
    return confidnet, optimizer, epoch_loss


def training_loop_confidnet(
    confidnet, classifier, 
    criterion, optimizer, scheduler, 
    train_loader, valid_loader, epochs, device, 
    num_heads=-1, supcon_classifier=True, supcon_confid_loss=None, 
    file_name='model.pth', print_every=1
):
 
    best_loss = 1e10
    best_acc = 0
    train_losses = []
    valid_losses = []
    
    freeze_parameters(classifier)
    supcon_confid = supcon_confid_loss != None
    
    if supcon_confid:
        train_loader.dataset.set_transform(TwoCropTransform(train_loader.dataset.transforms))
    
    for epoch in range(0, epochs):
        # training
        confidnet, optimizer, train_loss = train_rnn(
            confidnet, classifier, train_loader, 
            criterion, optimizer, num_heads, device, 
            supcon_classifier=supcon_classifier, supcon_confid_loss=supcon_confid_loss
        )
        
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            confidnet, valid_loss = validate_rnnconfidnet(
                confidnet, classifier, valid_loader, criterion, device, 
                multioutput=True, supcon_classifier=supcon_classifier, supcon_confid=supcon_confid
            )
            
            valid_losses.append(valid_loss)
            
            if scheduler != None:
                scheduler.step(valid_loss)
       
        if (valid_loss < best_loss):
            torch.save(confidnet.state_dict(), file_name)
            best_loss = valid_loss
    
        if epoch % print_every == (print_every - 1):                
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t')

    #plot_losses(train_losses, valid_losses)
    
    return confidnet, (train_losses, valid_losses)

