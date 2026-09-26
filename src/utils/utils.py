import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score


import os
import random
import math
import seaborn as sns
from datetime import datetime


def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    
    
def choose_optimizer(model, optimizer_name, lr=1e-3, wd=1e-5):
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unrecognized optimizer_name '{optimizer_name}'. Expected one of: 'Adam', 'AdamW', 'SGD'.")

    return optimizer

    
def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
    
    
def img_to_patch(x, patch_size=4, denoising_filter=None, flatten_channels=True, visualize=False):
    """
    Source: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html
    
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)              # [B, H'*W', C, p_H, p_W]
    
    if flatten_channels:
        x = x.flatten(2, 4)          # [B, H'*W', C*p_H*p_W]
        
    if visualize:
        #img_patches = img_to_patch(x, patch_size=4, flatten_channels=False)

        fig, ax = plt.subplots(x.shape[0], 1, figsize=(14,3))
        fig.suptitle("Images as input sequences of patches")
        for i in range(x.shape[0]):
            img_grid = torchvision.utils.make_grid(x[i], nrow=64, normalize=True, pad_value=0.9)
            img_grid = img_grid.permute(1, 2, 0)
            ax[i].imshow(img_grid)
            ax[i].axis('off')
        plt.show()
        plt.close()
        
    return x  

def patch_to_img(x, patch_size=4):
    B, S, L = x.shape
    
    x = x.unflatten(dim=-1, sizes=(1, patch_size, patch_size) )
    size = int(math.sqrt(S))
    x = x.unflatten(dim=1, sizes=(size, size))
    x = x.permute(0, 3, 1, 4, 2, 5)
    B, C, H_, _, W_, _  = x.shape       
    x = x.reshape(B, C, H_ * patch_size, W_ * patch_size)
    
    return x


def accuracy_cnn(model, data_loader, device, supcon=False):
    correct_preds = 0 
    n = 0
    idx = []   
        
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            y_true = y_true.to(device)
            
            if supcon:
                y_preds, _ = model(X)
            else:
                y_preds = model(X)
                
            n += y_true.size(0)
            #correct_preds += torch.div((1*(torch.sigmoid(y_preds) >= 0.5) == y_true).sum(dim=1), y_true.size(dim=-1), 
            #                           rounding_mode='trunc').float().sum()
            
            correct_preds += ((1*(torch.sigmoid(y_preds) >= 0.5) == y_true).sum(dim=1) == y_true.size(dim=-1)).float().sum()
  
    torch.cuda.empty_cache()
    return (correct_preds / n).item()

def f1score_cnn(model, data_loader, device, supcon=False):
    f1 = 0 
    n = 0
    
    y_preds = []
    labels = []
       
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            labels.append( y_true.detach().cpu().numpy() )   
            
            if supcon:
                y_hat, _ = model(X)
            else:
                y_hat = model(X)
                
            if y_true.dim() > 1:
                y_hat = (1*(torch.sigmoid(y_hat) >= 0.5)).detach().cpu().numpy()
            else:
                y_hat = (y_hat.argmax(dim=-1)).detach().cpu().numpy()
                
            y_preds.append( y_hat )
            n += y_true.size(0)
        
        if y_true.dim() == 1:
            labels = np.concatenate(labels) 
            y_preds = np.concatenate(y_preds) 
        else:
            labels =  np.concatenate(labels) 
            y_preds =  np.concatenate(y_preds)
            
        f1 = f1_score(labels, y_preds, average='macro')

    return f1


def validate_cnn(model, valid_loader, criterion, device):
    '''
    Function for the validation step of the training loop
    '''
   
    with torch.no_grad():
        model.eval()
        running_loss = 0

        for X, y_true in valid_loader:

            X = X.to(device)
            y_true = y_true.to(device)

            # Forward pass and record loss
            y_preds = model(X)
            loss = criterion(y_preds, y_true) #.sum(dim=1).mean() 

            running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
    torch.cuda.empty_cache()    
    return model, epoch_loss


def accuracy_rnn(model, data_loader, device, patch_order=None, prediction_index=-1, offset1=0, offset2=256, repeat=0, ensemble=False, supcon=False):
    correct_preds = 0 
    n = 0
    idx = []   
        
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:
            
            X = X.to(device)
            y_true = y_true.to(device)
            
            if supcon:
                y_preds, _, _ = model(X, patch_order=patch_order, prediction_index=prediction_index)
            else:
                y_preds, _ = model(X, patch_order=patch_order, prediction_index=prediction_index)
           
            n += y_true.size(0)
            
            if model.dataset_name == "mixedwm38":
                correct_preds += ((1*(torch.sigmoid(y_preds) >= 0.5) == y_true).sum(dim=1) == y_true.size(dim=-1)).float().sum()
            else:
                correct_preds += (y_preds.argmax(dim=1) == y_true).float().sum()
            
    torch.cuda.empty_cache()
    return (correct_preds/n).item()


def validate_rnn(model, valid_loader, criterion, device, patch_order=None, multioutput=False, supcon_loss=None):
    '''
    Function for the validation step of the training loop
    '''
    with torch.no_grad():
        model.eval()
        running_loss = 0

        for X, y_true in valid_loader:

            X = X.to(device)
            y_true = y_true.to(device)

            # Forward pass and record loss
            if supcon_loss == None:
                if patch_order != None:
                    y_preds, _ = model(X, patch_order=patch_order, multioutput=multioutput)
                else:    
                    y_preds, _ = model(X, multioutput=multioutput)
            else:
                if patch_order != None:
                    y_preds, _, _ = model(X, patch_order=patch_order, multioutput=multioutput)
                else:    
                    y_preds, _, _ = model(X, multioutput=multioutput)                      

            if multioutput: 
                loss  = []
                for i in range(len(y_preds)):
                    loss_head = criterion(y_preds[i], y_true).sum(dim=1).mean() 
                    loss.append(loss_head)

                loss = torch.stack(loss, dim=0)
                loss = loss.min(dim=0)[0].mean()
            else:
                loss = criterion(y_preds, y_true).sum(dim=1).mean()  

            running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
    torch.cuda.empty_cache()    
    
    return model, epoch_loss


def validate_rnnconfidnet(confidnet, classifier, valid_loader, criterion, device, patch_order=None, multioutput=False, supcon_classifier=False, supcon_confid=False):
    '''
    Function for the validation step of the training loop
    '''
    
    with torch.no_grad():
        confidnet.eval()
        classifier.eval()
        running_loss = 0

        for X, y_true in valid_loader:

            X = X.to(device)
            y_true = y_true.to(device)

            # Forward pass and record loss
            if patch_order != None:
                if supcon_classifier:
                    y_preds, hidden, _ = classifier(X, patch_order=patch_order, multioutput=multioutput)
                else:    
                    y_preds, hidden = classifier(X, patch_order=patch_order, multioutput=multioutput)
            else:  
                if supcon_classifier:
                    y_preds, hidden, _ = classifier(X, multioutput=multioutput)
                else:
                    y_preds, hidden = classifier(X, multioutput=multioutput)

            if multioutput: 
                hidden = torch.stack(hidden, dim=0)
                #hidden = hidden.reshape(hidden.shape[2], hidden.shape[1], hidden.shape[0] * hidden.shape[-1]) ##############
                y_preds = torch.stack(y_preds, dim=0)
                
                if supcon_confid:
                    confidence, _, _ = confidnet(hidden.transpose(0, 1), multioutput=True)
                else:
                    confidence, _ = confidnet(hidden.transpose(0, 1), multioutput=True)
                    
                confidence = torch.stack(confidence, dim=0)

                loss_conf = criterion(confidence, y_preds, y_true)   
                loss = loss_conf.mean()
            else:
                confidence = confidnet(hidden[-1])
                loss = criterion(confidence, y_preds, y_true)

            running_loss += loss.item() * X.size(0)

        epoch_loss = running_loss / len(valid_loader.dataset)
        
    return confidnet, epoch_loss


def accuracy_rnnconfidnet(
    model, confidnet, data_loader, device, 
    cnn_model=None, patch_order=None, prediction_index=-1, 
    offset1=0, offset2=169, supcon_classifier=False, supcon_confid=False
):
    model.eval()
    confidnet.eval()
    if cnn_model != None:
        cnn_model.eval()
        
    correct_preds = 0 
    n = 0
    idx = []   
    best_heads = []
    num_scans = []
    
    all_preds = []
    all_labels = []
    all_confid = []
        
    with torch.no_grad():
        model.eval()
        #encoder.eval()
        confidnet.eval()
        for i, (X, y_true) in enumerate(data_loader):

            X = X.to(device)
            y_true = y_true.to(device)
           
            if prediction_index in ['best-e', 'best-max', 'best-5', 'min-diff', 'confidnet']:
                if supcon_classifier:
                    y_preds, hidden, _ = model(X, patch_order=patch_order, multioutput=True)
                else:
                    y_preds, hidden = model(X, patch_order=patch_order, multioutput=True)
                    
                y_preds = torch.stack(y_preds, dim=1)
                
                hidden = torch.stack(hidden, dim=0) 
                #hidden = hidden.reshape(hidden.shape[2], hidden.shape[1], hidden.shape[0] * hidden.shape[-1])
                
                if supcon_confid:
                    confidence, _, _ = confidnet(hidden.transpose(0, 1), multioutput=True)
                else:
                    confidence, _ = confidnet(hidden.transpose(0, 1), multioutput=True)
                
                
                #confidence, _ = confidnet(X, patch_order=classifier.patch_order, multioutput=True)
                    
                confidence = torch.stack(confidence, dim=0)
                confidence = confidence.transpose(0, 1).squeeze(-1)
                
                #confidence = (torch.sigmoid(y_preds).transpose(0, 1) * y_true + (1 - torch.sigmoid(y_preds).transpose(0, 1)) * (1 - y_true)).transpose(0, 1) 
                if y_true.dim() == 1:
                    head_mask = (confidence[:, offset1:offset2] >= 0.5).sum(dim=-1)
                    best_head = (1*(confidence[:, offset1:offset2] >= 0.5)).argmax(dim=-1) + offset1
                    best_head[head_mask == 0] = confidence[head_mask == 0, offset1:offset2].argmax(dim=-1) + offset1
                else:
                    head_mask = ((confidence[:, offset1:offset2] >= 0.5).sum(dim=-1) == y_true.size(dim=-1)).sum(dim=-1)
                    best_head = (1*((confidence[:, offset1:offset2] >= 0.5).sum(dim=-1) == y_true.size(dim=-1)) ).argmax(dim=-1) + offset1
                                
                idx.append(best_head.clone())
                best_head[head_mask == 0] = offset2 - 1
                num_scans.append(best_head.clone() + 1)                
                #print(best_head)
                #y_preds = torch.stack([y_preds[i, idx[-1][i], :] for i in range(len(idx[-1]))], dim=0)
                y_preds = torch.stack([y_preds[i, best_head[i], :] for i in range(len(best_head))], dim=0)
            
            else:
                if supcon_classifier:
                    y_preds, _, _ = model(X, patch_order=patch_order, prediction_index=prediction_index)
                else:
                    y_preds, _ = model(X, patch_order=patch_order, prediction_index=prediction_index)
           
            if cnn_model != None:
                y_preds[head_mask == 0] = cnn_model(X[head_mask == 0])
                
            n += y_true.size(0)
            if y_true.dim() > 1:
                correct_preds += ((1*(torch.sigmoid(y_preds) >= 0.5) == y_true).sum(dim=1) == y_true.size(dim=-1)).float().sum()
            else:
                correct_preds += (y_preds.argmax(dim=1) == y_true).float().sum()
            
            all_labels.append(y_true)
            all_preds.append(y_preds)
            
            
    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0).argmax(dim=-1)
    
    #f1 = f1_score(all_labels.detach().cpu().numpy(), all_preds.detach().cpu().numpy(), average='macro')        
            
    if prediction_index in ['best-e', 'best-max', 'best-5', 'min-diff', 'confidnet', 'confid']:
        idx = torch.cat(idx, dim=0)
        num_scans = torch.cat(num_scans, dim=0)
        return (correct_preds/n).item(), idx, num_scans
    
    torch.cuda.empty_cache()
    #return (correct_preds/n).item(), f1.item(), prediction_index, num_scans
    return (correct_preds/n).item(), prediction_index, num_scans