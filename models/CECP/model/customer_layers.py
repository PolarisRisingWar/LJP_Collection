# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np


# -------------------------------------------------------------------------------------------------------------
class AggregationLayerPretrain(nn.Module):
    def __init__(self, config):
        super(AggregationLayerPretrain, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
        self.reset_parameters()
        
    def forward(self, fact, elements_p):
        fact_Q, _ = fact.max(dim=1)
        
        f2e = torch.matmul(fact_Q, self.weights)
        f2e = torch.matmul(f2e, elements_p.T)
        f2e = nn.functional.softmax(f2e, dim=-1)
        f2e = torch.unsqueeze(f2e, dim=-1)
        elements_p_ = torch.multiply(torch.unsqueeze(elements_p, dim=0), f2e)   
        elements_p_ = elements_p_.sum(dim=1)    # [1, d]
            
        e2f = torch.matmul(fact, self.weights)
        e2f = torch.multiply(e2f, torch.unsqueeze(elements_p_, dim=1))
        e2f = torch.sum(e2f, dim=-1, keepdims=True)
        e2f = nn.functional.softmax(e2f, dim=1)
        fact_ = torch.multiply(fact, e2f)
        fact_ = fact_.sum(dim=1)
        
        return torch.cat([fact_, elements_p_, torch.multiply(fact_, elements_p_)], dim=1), elements_p_
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5))

# --------------------------------------------------------------------------------------------------------------
class AggregationLayer(nn.Module):
    def __init__(self, config):
        super(AggregationLayer, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
        self.reset_parameters()
        
    def forward(self, fact, elements_p):  
        fact_Q, _ = fact.max(dim=0, keepdims=True)
        
        f2e = torch.matmul(fact_Q, self.weights)
        f2e = torch.matmul(f2e, elements_p.T)
        f2e = nn.functional.softmax(f2e, dim=-1)
        elements_p_ = torch.multiply(elements_p, f2e.T)   
        elements_p_ = elements_p_.sum(dim=0, keepdims=True)    # [1, d]
            
        e2f = torch.matmul(fact, self.weights)
        e2f = torch.matmul(e2f, elements_p_.T)
        e2f = nn.functional.softmax(e2f, dim=0)
        self.e2f = e2f
        fact_ = torch.multiply(fact, e2f)
        fact_ = fact_.sum(dim=0, keepdims=True)
        
        return fact_, elements_p_
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5))
        
        
# -------------------------------------------------------------------------------------------------------------------------------
def PFILayer(elements_p):   # [149,d]
    std = elements_p.std(dim=0).square()
    u = 2. * torch.sigmoid(-std)
            
    nonPFI_elements = torch.multiply(u, elements_p)
    nonPFI_elements = nonPFI_elements.mean(dim=0, keepdims=True)

    return torch.clamp(elements_p - nonPFI_elements, -1.0, 1.0)


# ----------------------------------------------------------------------------------------------------------------------------------
class ActionLayer(nn.Module):
    def __init__(self, config):
        super(ActionLayer, self).__init__()
        self.config = config
        self.weights = nn.ParameterDict({'W_pointer_e':nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))})
        self.weights.update({'W_pointer_h':nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))})
        self.gate_layer = nn.Linear(config.hidden_size*3, 1)
        self.reset_parameters()

    def forward(self, fact, context_e, h, one_sample_sent_num, S_row):  
        mask_multiply = 1 - S_row
        mask_add = S_row * -1e10
        if one_sample_sent_num < self.config.sent_num:
            mask_add[-(self.config.sent_num - one_sample_sent_num):] = -1e10
            mask_multiply[-(self.config.sent_num - one_sample_sent_num):] = 0
        
        f_w = torch.matmul(fact, self.weights['W_pointer_e'])
        f_w_e = torch.matmul(f_w, context_e.T)
        
        f_w = torch.matmul(fact, self.weights['W_pointer_h'])
        f_w_h = torch.matmul(f_w, h.T)    # shape=[n, 1]
    
        gate = self.gate_layer(torch.cat([fact, h.expand(fact.shape[0], -1), context_e.expand(fact.shape)], dim=1))
        gate = torch.sigmoid(gate)    # [n, 1]
        distribution = torch.multiply(f_w_e, gate) + torch.multiply(f_w_h, 1.0-gate)

        
        distribution = torch.multiply(distribution, torch.from_numpy(mask_multiply)) + torch.from_numpy(mask_add)
        distribution = nn.functional.softmax(distribution, dim=0)    # [n, 1]

        return distribution.T   # [1, n]

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights['W_pointer_e'], a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.weights['W_pointer_h'], a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.gate_layer.weight, a=np.sqrt(5))
        nn.init.constant_(self.gate_layer.bias, 0.)
        
        
        
        