import torch
import torch.nn as nn
import torch.nn.functional as F



class RNNSelfConfidMSELoss(nn.modules.loss._Loss):
    def __init__(self, num_classes=8, weight=2):
        super().__init__()
        
        self.num_classes = num_classes
        self.weight = weight
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, confidence, y_preds, target):
        if target.dim() == 1:
            probs = F.softmax(y_preds, dim=-1)
            weights = torch.ones((y_preds.size(0), y_preds.size(1))).type(torch.FloatTensor).to(self.device)
            weights[(probs.argmax(dim=-1) != target)] *= self.weight
            labels_hot = F.one_hot(target, self.num_classes).unsqueeze(0).repeat(y_preds.size(0), 1, 1).to(self.device)

            loss = weights * (confidence.squeeze(-1) - (probs * labels_hot).sum(dim=-1)) ** 2
        else:
            probs = torch.sigmoid(y_preds)
            weights = torch.ones_like(y_preds).type(torch.FloatTensor).to(self.device)

            confidence_true = (probs * target + (1 - probs) * (1 - target))
            weights[(1*(probs >= 0.5) != target)] *= self.weight 
            #weights[(1*(probs >= 0.5) != target)] *= torch.exp( self.weight * (probs[(1*(probs >= 0.5) != target)] - 0.5).abs() )

            loss = ( weights * (confidence - confidence_true) ** 2).mean(dim=-1)
        
        return torch.mean(loss, dim=1)

    
class SelfConfidTCPRLoss(nn.modules.loss._Loss):
    def __init__(self, num_classes=8, weight=1):
        super().__init__()
        
        self.num_classes = num_classes
        self.weight = weight
        self.device = device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, confidence, y_preds, target):
        probs = torch.sigmoid(y_preds)
        maxprob = probs.max(dim=-1)[0].unsqueeze(dim=-1)
        
        weights = torch.ones_like(y_preds).type(torch.FloatTensor).to(self.device)
        confidence_true = (probs * target + (1 - probs) * (1 - target)) 
        
        weights[(1*(probs >= 0.5) != target)] *= self.weight
        #labels_hot = F.one_hot(target, self.num_classes).to(self.device)
        
        loss = (weights * (confidence_true - confidence / maxprob) ** 2).mean(dim=-1)
        #loss = weights * (confidence - (probs * labels_hot).sum(dim=1) / maxprob) ** 2
        
        return torch.mean(loss)