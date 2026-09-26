import os
import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F

from utils.utils import img_to_patch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def eval_policy_ood(agent, env, train=False, confidnet=None, cnn_model=None, offset=0, ensemble=False):
    if ensemble:
        for i in range(len(agent.actors)):
            agent.actors[i].eval()
    else:
        agent.actor.eval()
    
    if cnn_model != None:
        cnn_model.eval()
    
    if train:
        data_loader = env.train_loader
    else:
        data_loader = env.test_loader
    
    correct_preds = 0
    total_reward = 0
    gamma = agent.gamma
    
    if ensemble:
        num_classes = env.classifier[0].num_classes
    else:
        num_classes = env.classifier.num_classes
    
    patch_order = []
    stopping = []
    max_confids = []
    
    correct_full = 0
    correct_notfull = 0
    
    preds = []
    labels = []
    
    for batch_idx, (X, y) in enumerate(data_loader):
        X = X.to(device)
        y = y.to(device)
        
        patches = img_to_patch(X, patch_size=env.patch_size)
        
        batch_size = patches.size(0)
        seq_len = patches.size(1)
        
        _, observation = env.reset(patches, y)
        
        if ensemble:
            if env.classifier[0].cell_type == "LSTM":
                observation = [observation[i][0] for i in range(len(observation))]
        else:
            if env.classifier.cell_type == "LSTM":
                observation = observation[0]    
        
        discount = 1.0
        patch_order_batch = []
        agent.memory.clear_memory()
        
        if confidnet != None:
            if ensemble:
                hidden_dim = confidnet[0].hidden_dim
            else:
                hidden_dim = confidnet.hidden_dim
                
            filled = torch.zeros(batch_size).to(device)
            is_anomaly = torch.zeros(batch_size, dtype=torch.long).to(device)
            stopping_batch = torch.zeros(batch_size).to(device)
                    
        for t in range(0, seq_len):
            actions_mask = env.action_mask.clone()
            action, prob = agent.choose_action(observation, actions_mask, determenistic=True)

            if confidnet != None and t >= offset:
                outputs, observation_, reward, done, info = env.step(action, mode='inference')
                confidence = reward.to(device)
                
                if not ensemble:
                    outputs = torch.sigmoid(outputs)
                
                anomaly = (outputs >= 0.5).sum(dim=-1) > 0
                conf_score = (confidence >= 0.5).sum(dim=-1) 
                criterion = (anomaly == False)*(conf_score == outputs.size(dim=-1)) + (((confidence >= 0.5)*(outputs >= 0.5)).sum(dim=-1) > 0) > 0
                is_anomaly[criterion * (filled == 0)] = 1*(((outputs >= 0.5)[criterion * (filled == 0)]).sum(dim=-1) > 0)
                
                stopping_batch[criterion * (filled == 0)] = t + 1
                filled[criterion] = 1
            else:
                y_pred, observation_, reward, done, info = env.step(action)
        
            
            if env.curr_time >= env.start_policy:
                total_reward += (discount * reward).sum()
                discount *= gamma
            
            observation = observation_
            patch_order_batch.append(action)
            
        #if not ensemble:
        #y_pred = torch.sigmoid(y_pred)
          
        if confidnet != None:  
            if cnn_model == None:
                is_anomaly[filled == 0] = 1*((outputs[filled == 0] >= 0.5).sum(dim=-1) > 0)
            else:
                if (filled == 0).sum() > 0:
                    is_anomaly[filled == 0] = 1*((torch.sigmoid(cnn_model(X[filled == 0])) >= 0.5).sum(dim=-1) > 0)
            
                
            stopping_batch[filled == 0] = seq_len
            stopping.append(stopping_batch)
            
            #correct_full += ((1*(y_pred[stopping_batch == seq_len] >= 0.5) == y[stopping_batch == seq_len]).sum(dim=1) == y.size(dim=-1)).float().sum()
            #correct_notfull += ((1*(y_pred[stopping_batch < seq_len] >= 0.5) == y[stopping_batch < seq_len]).sum(dim=1) == y.size(dim=-1)).float().sum()
            
        patch_order_batch = torch.stack(patch_order_batch, dim=1).detach().cpu()
        patch_order.append(patch_order_batch)
        correct_preds += (is_anomaly == (y.sum(dim=-1) > 0)).sum()
        preds.append(is_anomaly)
        labels.append(1*(y.sum(dim=-1) > 0))
        #print(is_anomaly)
        #print(1*(y.sum(dim=-1) > 0))
        print(batch_idx)       
    
    patch_order = torch.cat(patch_order, dim=0)
    preds = torch.cat(preds, dim=0).detach().cpu().numpy()
    labels = torch.cat(labels, dim=0).detach().cpu().numpy()
    
    acc = 100 * correct_preds / len(data_loader.dataset)
    f1 = f1_score(labels, preds, average='macro')
    #f1 = 0
    score = total_reward / len(data_loader.dataset)
    
    if confidnet != None:        
        stopping = torch.cat(stopping, dim=0)
        #print((stopping < seq_len).sum(),  (filled == 0).sum())
        #print(f"Full scan correct: {(100 * correct_full / (stopping == seq_len).sum()):.2f}, NOT Full scan correct: {(100 * correct_notfull / (stopping < seq_len).sum()):.2f}")
        return acc, f1, score, patch_order, stopping
    
    return acc, f1, score, patch_order, None


def predict_ood_policy(agent, env, train=False, confidnet=None, cnn_model=None, offset=0, ensemble=False):
    if ensemble:
        for i in range(len(agent.actors)):
            agent.actors[i].eval()
    else:
        agent.actor.eval()
    
    if cnn_model != None:
        cnn_model.eval()
    
    if train:
        data_loader = env.train_loader
    else:
        data_loader = env.test_loader
    
    correct_preds = 0
    total_reward = 0
    gamma = agent.gamma
    
    if ensemble:
        num_classes = env.classifier[0].num_classes
    else:
        num_classes = env.classifier.num_classes
   
    y_preds = []
    stopping = []
    patch_order = []
    
    correct_full = 0
    correct_notfull = 0
    
    for batch_idx, (X, y) in enumerate(data_loader):
        X = X.to(device)
        y = y.to(device)
        
        patches = img_to_patch(X, patch_size=env.patch_size)
        
        batch_size = patches.size(0)
        seq_len = patches.size(1)
        
        _, observation = env.reset(patches, y)
        
        if ensemble:
            if env.classifier[0].cell_type == "LSTM":
                observation = [observation[i][0] for i in range(len(observation))]
        else:
            if env.classifier.cell_type == "LSTM":
                observation = observation[0]    
        
        discount = 1.0
        patch_order_batch = []
        agent.memory.clear_memory()
        
        if confidnet != None:
            if ensemble:
                hidden_dim = confidnet[0].hidden_dim
            else:
                hidden_dim = confidnet.hidden_dim
                
            filled = torch.zeros(batch_size).to(device)
            y_pred = torch.zeros((batch_size, num_classes), dtype=torch.long).to(device)
            stopping_batch = torch.zeros(batch_size).to(device)
        
        for t in range(0, seq_len):
            actions_mask = env.action_mask.clone()
            action, prob = agent.choose_action(observation, actions_mask, determenistic=True)

            if confidnet != None and t >= offset:
                outputs, observation_, reward, done, info = env.step(action, mode='inference')
                confidence = reward.to(device)
                
                if not ensemble:
                    outputs = torch.sigmoid(outputs)
                    
                anomaly = (outputs >= 0.5).sum(dim=-1) > 0
                conf_score = (confidence >= 0.5).sum(dim=-1) 
                criterion = (anomaly == False)*(conf_score == outputs.size(dim=-1)) + (((confidence >= 0.5)*(outputs >= 0.5)).sum(dim=-1) > 0) > 0
                
                y_pred[criterion * (filled == 0)] = (1*(confidence >= 0.5)*(outputs >= 0.5))[criterion * (filled == 0)]
                stopping_batch[criterion * (filled == 0)] = t + 1
                filled[criterion] = 1
            else:
                y_pred, observation_, reward, done, info = env.step(action)
        
            if env.curr_time >= env.start_policy:
                total_reward += (discount * reward).sum()
                discount *= gamma 
            
            observation = observation_
            patch_order_batch.append(action)
            
        if confidnet != None:  
            if cnn_model == None:
                y_pred[filled == 0] = 1*(outputs[filled == 0] >= 0.5)
            else:
                if (filled == 0).sum() > 0:
                    y_pred[filled == 0] = 1*(torch.sigmoid(cnn_model(X[filled == 0])) >= 0.5)
                    
            stopping_batch[filled == 0] = seq_len
            stopping.append(stopping_batch)
            
            #correct_full += (y_pred[filled == 0].argmax(dim=-1) == y[filled == 0]).sum()
            #correct_notfull += (y_pred[filled == 1].argmax(dim=-1) == y[filled == 1]).sum()
            
        patch_order_batch = torch.stack(patch_order_batch, dim=1).detach().cpu()    
        patch_order.append(patch_order_batch)
        
        #stopping.append(stopping_batch)
        y_preds.append(y_pred)
        print(batch_idx)       
        
    y_preds = torch.cat(y_preds, dim=0)
    stopping = torch.cat(stopping, dim=0)
    patch_order = torch.cat(patch_order, dim=0)
    
    return y_preds, patch_order, stopping



