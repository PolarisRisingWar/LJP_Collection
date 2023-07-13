# -*- coding: utf-8 -*-
import torch
from torch import nn
import numpy as np
from customer_layers import AggregationLayer, ActionLayer, AggregationLayerPretrain
import copy


def aggregate_with_weights(fact, elements_p, aggregate_weight):    # [num_selected_sentences, d],  [149, d]  
    ''' for environment '''
    fact_Q, _ = fact.max(dim=0, keepdims=True)
    
    f2e = torch.matmul(fact_Q, aggregate_weight)
    f2e = torch.matmul(f2e, elements_p.T)
    f2e = nn.functional.softmax(f2e, dim=-1)
    elements_p_ = torch.multiply(elements_p, f2e.T)   
    elements_p_ = elements_p_.sum(dim=0, keepdims=True)    # [1, d]
        
    e2f = torch.matmul(fact, aggregate_weight)
    e2f = torch.matmul(e2f, elements_p_.T)  
    e2f = nn.functional.softmax(e2f, dim=0)
    fact_ = torch.multiply(fact, e2f)
    fact_ = fact_.sum(dim=0, keepdims=True)
    
    return fact_, elements_p_



# --------------------------------------------------------------------------------------------------------------------------------
class Environment():
    def reset(self, fact, S, elements, aggregate_weights):
        self.fact = fact
        self.S = S
        
        fact_0_selected = torch.zeros([1, 128])
        fact_1_selected = torch.zeros([1, 128])
        fact_2_selected = torch.zeros([1, 128])
        fact_3_selected = torch.zeros([1, 128])

        self.selected_sente = [fact_0_selected, fact_1_selected, fact_2_selected, fact_3_selected]
        
        fact_0_context, context_e_0 = aggregate_with_weights(fact_0_selected, elements[0], aggregate_weights[0])
        fact_1_context, context_e_1 = aggregate_with_weights(fact_1_selected, elements[1], aggregate_weights[1])
        fact_2_context, context_e_2 = aggregate_with_weights(fact_2_selected, elements[2], aggregate_weights[2])
        fact_3_context, context_e_3 = aggregate_with_weights(fact_3_selected, elements[3], aggregate_weights[3])
        self.context_e = [context_e_0, context_e_1, context_e_2, context_e_3]
        self.context_f = [fact_0_context, fact_1_context, fact_2_context, fact_3_context]
        
        return self.selected_sente
    
    def step(self, action, p, elements_p, aggregate_weights):
        self.S[p, action] = 1
        fact_p_selected   = self.fact[p][self.S[p, :]==1, :]
        self.selected_sente[p] = fact_p_selected
        
        context_f_p, context_e_p = aggregate_with_weights(fact_p_selected, elements_p, aggregate_weights[p])
        self.context_f[p] = context_f_p
        self.context_e[p] = context_e_p
        
        return self.selected_sente


class PredNetPretrain(nn.Module):
    def __init__(self, config):
        super(PredNetPretrain, self).__init__()
        self.aggregation_layer_0 = AggregationLayerPretrain(config)
        self.aggregation_layer_1 = AggregationLayerPretrain(config)
        self.aggregation_layer_2 = AggregationLayerPretrain(config)
        self.aggregation_layer_3 = AggregationLayerPretrain(config)
        self.dense_pred = nn.Linear(config.hidden_size*12, config.num_charges)
        
    def forward(self, fact, elements):
        fact_elements_0, elements_p_0 = self.aggregation_layer_0(fact[0], elements[0])
        fact_elements_1, elements_p_1 = self.aggregation_layer_1(fact[1], elements[1])
        fact_elements_2, elements_p_2 = self.aggregation_layer_2(fact[2], elements[2])
        fact_elements_3, elements_p_3 = self.aggregation_layer_3(fact[3], elements[3])

        concat_f_e = torch.cat([fact_elements_0, fact_elements_1, fact_elements_2, fact_elements_3], dim=1)
        pred = self.dense_pred(concat_f_e)
        return pred


# --------------------------------------------------------------------------------------------------------------------------------
class PredNet(nn.Module):
    def __init__(self, config):
        super(PredNet, self).__init__()
        self.aggregation_layer_0 = AggregationLayer(config)
        self.aggregation_layer_1 = AggregationLayer(config)
        self.aggregation_layer_2 = AggregationLayer(config)
        self.aggregation_layer_3 = AggregationLayer(config)
        self.dense_pred = nn.Linear(config.hidden_size*12, config.num_charges)
        
    def forward(self, fact, elements, batch_S):
        reinforced_data = []
        self.e2f = []
        
        for i in range(len(batch_S)):
            tmp_S = batch_S[i]
            tmp_fact = [fact[0][i], fact[1][i], fact[2][i], fact[3][i]]
            data = self.aggregate_e_p(tmp_fact, elements, tmp_S.astype(np.float32))
            reinforced_data.append(data)
                    
        f_e = torch.cat(reinforced_data, dim=0)
        pred = self.dense_pred(f_e)
        
        return pred
        
    def aggregate_e_p(self, f, elements, S):
        fact_0_selected = f[0][S[0, :]==1, :]
        fact_1_selected = f[1][S[1, :]==1, :]
        fact_2_selected = f[2][S[2, :]==1, :]
        fact_3_selected = f[3][S[3, :]==1, :]
        
        fact_0_context, context_e_0 = self.aggregation_layer_0(fact_0_selected, elements[0])  #[1,128]  [1,128]  [1,149]
        fact_1_context, context_e_1 = self.aggregation_layer_1(fact_1_selected, elements[1])
        fact_2_context, context_e_2 = self.aggregation_layer_2(fact_2_selected, elements[2])
        fact_3_context, context_e_3 = self.aggregation_layer_3(fact_3_selected, elements[3])
        self.e2f.append([copy.deepcopy(self.aggregation_layer_0.e2f.detach().cpu().numpy()),
                         copy.deepcopy(self.aggregation_layer_1.e2f.detach().cpu().numpy()),
                         copy.deepcopy(self.aggregation_layer_2.e2f.detach().cpu().numpy()),
                         copy.deepcopy(self.aggregation_layer_3.e2f.detach().cpu().numpy())])
        
        return torch.cat([fact_0_context, context_e_0, torch.multiply(fact_0_context, context_e_0),
                          fact_1_context, context_e_1, torch.multiply(fact_1_context, context_e_1),
                          fact_2_context, context_e_2, torch.multiply(fact_2_context, context_e_2),
                          fact_3_context, context_e_3, torch.multiply(fact_3_context, context_e_3)], dim=1)    # [4,149]

    
# ----------------------------------------------------------------------------------------------------------------------------------
class ACNet(nn.Module):
    def __init__(self, config):
        super(ACNet, self).__init__()
        self.config = config
        self.aggregation_layer_0 = AggregationLayer(config)
        self.aggregation_layer_1 = AggregationLayer(config)
        self.aggregation_layer_2 = AggregationLayer(config)
        self.aggregation_layer_3 = AggregationLayer(config)
        
        self.action_layer = ActionLayer(config)
        self.dense_h = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dense_value_1 = nn.Linear(config.hidden_size*4, 256)
        self.dense_value_2 = nn.Linear(256, 1)

        self.reset_parameters()
        
    def forward(self, selected_sente_p, fact, elements_p, p, one_sample_sent_num, S_row=None):  
        if p==0:
            context_f_p, context_e_p = self.aggregation_layer_0(selected_sente_p, elements_p)
        elif p==1:
            context_f_p, context_e_p = self.aggregation_layer_1(selected_sente_p, elements_p)
        elif p==2:
            context_f_p, context_e_p = self.aggregation_layer_2(selected_sente_p, elements_p)
        elif p==3:
            context_f_p, context_e_p = self.aggregation_layer_3(selected_sente_p, elements_p)
        self.context_f[p] = context_f_p
        
        self.h_tmp = self.dense_h(torch.cat([self.h, context_f_p], dim=1))
        self.h_tmp = torch.tanh(self.h_tmp)
        self.h = self.h_tmp.detach()
        
        distribution = self.action_layer(fact, context_e_p, self.h_tmp, one_sample_sent_num, S_row)  
        
        value = self.dense_value_1(torch.cat([self.context_f[0], self.context_f[1], self.context_f[2], self.context_f[3]], dim=1))
        value = torch.tanh(value)
        value = self.dense_value_2(value)
        
        return distribution, value
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.dense_h.weight, a=np.sqrt(5))
        nn.init.constant_(self.dense_h.bias, 0.)
        nn.init.kaiming_uniform_(self.dense_value_1.weight, a=np.sqrt(5))
        nn.init.constant_(self.dense_value_1.bias, 0.)
    
    def customer_init(self, selected_sente, elements):   
        self.context_f = [None] * 4
        self.context_f[0] = torch.zeros([1, 128])
        self.context_f[1] = torch.zeros([1, 128])
        self.context_f[2] = torch.zeros([1, 128])
        self.context_f[3] = torch.zeros([1, 128])
        
        self.h = torch.zeros(size=(1, self.config.hidden_size))

# -------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    pass
   









