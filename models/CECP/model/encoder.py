# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np
from customer_layers import PFILayer
import copy


# ---------------------------------------------------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, config, word_embedding):
        super(Encoder, self).__init__()
        self.config = config
        self.word_embedding = torch.tensor(word_embedding, requires_grad=False).to('cuda:0')
        word_embedding_size = 200

        self.lstm_0 = nn.GRU(input_size=word_embedding_size, hidden_size=int(self.config.hidden_size), batch_first=True,
                              num_layers=1, bidirectional=False)
        self.lstm_1 = nn.GRU(input_size=word_embedding_size, hidden_size=int(self.config.hidden_size), batch_first=True,
                             num_layers=1, bidirectional=False)
        self.lstm_2 = nn.GRU(input_size=word_embedding_size, hidden_size=int(self.config.hidden_size), batch_first=True,
                              num_layers=1, bidirectional=False)
        self.lstm_3 = nn.GRU(input_size=word_embedding_size, hidden_size=int(self.config.hidden_size), batch_first=True,
                              num_layers=1, bidirectional=False)
        
        self.lstm_e_0 = nn.GRU(input_size=word_embedding_size, hidden_size=int(self.config.hidden_size), batch_first=True,
                                num_layers=1, bidirectional=False)
        self.lstm_e_1 = nn.GRU(input_size=word_embedding_size, hidden_size=int(self.config.hidden_size), batch_first=True,
                                num_layers=1, bidirectional=False)
        self.lstm_e_2 = nn.GRU(input_size=word_embedding_size, hidden_size=int(self.config.hidden_size), batch_first=True,
                                num_layers=1, bidirectional=False)
        self.lstm_e_3 = nn.GRU(input_size=word_embedding_size, hidden_size=int(self.config.hidden_size), batch_first=True,
                                num_layers=1, bidirectional=False)
        
        self.dense_K = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.dense_K_e = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.square_d = np.sqrt(128.0)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, fact, elements):
        fact = torch.reshape(fact, shape=[-1, 32])
        fact = torch.nn.functional.embedding(fact, self.word_embedding)
        fact = self.dropout(fact)
        self.self_att = []
        
        d, _ = self.lstm_0(fact)
        Q = nn.MaxPool2d(kernel_size=[d.shape[1],1], stride=[1,1])(torch.unsqueeze(d, dim=1))
        Q = torch.squeeze(Q, dim=1)
        K = self.dense_K(d)
        V = d
        QK = nn.functional.softmax(torch.sum(torch.multiply(Q, K), dim=-1, keepdims=True) / self.square_d, dim=1)
        self.self_att.append(copy.deepcopy(np.reshape(QK.detach().cpu().numpy(), [-1, 64, 32])))   # [bs, 64, 32]
        d = torch.sum(torch.multiply(QK, V), dim=1)
        fact_0 = torch.reshape(d, shape=[-1, self.config.sent_num, d.shape[-1]])
        
        d, _ = self.lstm_1(fact) 
        Q = nn.MaxPool2d(kernel_size=[d.shape[1],1], stride=[1,1])(torch.unsqueeze(d, dim=1))
        Q = torch.squeeze(Q, dim=1)
        K = self.dense_K(d)
        V = d
        QK = nn.functional.softmax(torch.sum(torch.multiply(Q, K), dim=-1, keepdims=True) / self.square_d, dim=1)
        self.self_att.append(copy.deepcopy(np.reshape(QK.detach().cpu().numpy(), [-1, 64, 32])))
        d = torch.sum(torch.multiply(QK, V), dim=1)
        fact_1 = torch.reshape(d, shape=[-1, self.config.sent_num, d.shape[-1]])
        
        d, _ = self.lstm_2(fact)
        Q = nn.MaxPool2d(kernel_size=[d.shape[1],1], stride=[1,1])(torch.unsqueeze(d, dim=1))
        Q = torch.squeeze(Q, dim=1)
        K = self.dense_K(d)
        V = d
        QK = nn.functional.softmax(torch.sum(torch.multiply(Q, K), dim=-1, keepdims=True) / self.square_d, dim=1)
        self.self_att.append(copy.deepcopy(np.reshape(QK.detach().cpu().numpy(), [-1, 64, 32])))
        d = torch.sum(torch.multiply(QK, V), dim=1)
        fact_2 = torch.reshape(d, shape=[-1, self.config.sent_num, d.shape[-1]])
        
        d, _ = self.lstm_3(fact)   
        Q = nn.MaxPool2d(kernel_size=[d.shape[1],1], stride=[1,1])(torch.unsqueeze(d, dim=1))
        Q = torch.squeeze(Q, dim=1)
        K = self.dense_K(d)
        V = d
        QK = nn.functional.softmax(torch.sum(torch.multiply(Q, K), dim=-1, keepdims=True) / self.square_d, dim=1)
        self.self_att.append(copy.deepcopy(np.reshape(QK.detach().cpu().numpy(), [-1, 64, 32])))
        d = torch.sum(torch.multiply(QK, V), dim=1)
        fact_3 = torch.reshape(d, shape=[-1, self.config.sent_num, d.shape[-1]])
        
        
        # ----------------------------------------------------------------------------------------------------------------------
        d = torch.nn.functional.embedding(elements[0], self.word_embedding)
        d, _ = self.lstm_e_0(d)
        Q = nn.MaxPool2d(kernel_size=[d.shape[1],1], stride=[1,1])(torch.unsqueeze(d, dim=1))
        Q = torch.squeeze(Q, dim=1)
        K = self.dense_K_e(d)
        V = d
        QK = nn.functional.softmax(torch.sum(torch.multiply(Q, K), dim=-1, keepdims=True) / self.square_d, dim=1)
        elements_0 = torch.sum(torch.multiply(QK, V), dim=1)
        elements_0 = PFILayer(elements_0)

        d = torch.nn.functional.embedding(elements[1], self.word_embedding)
        d, _ = self.lstm_e_1(d)
        Q = nn.MaxPool2d(kernel_size=[d.shape[1],1], stride=[1,1])(torch.unsqueeze(d, dim=1))
        Q = torch.squeeze(Q, dim=1)
        K = self.dense_K_e(d)
        V = d
        QK = nn.functional.softmax(torch.sum(torch.multiply(Q, K), dim=-1, keepdims=True) / self.square_d, dim=1)
        elements_1 = torch.sum(torch.multiply(QK, V), dim=1)
        elements_1 = PFILayer(elements_1)
            
        d = torch.nn.functional.embedding(elements[2], self.word_embedding)
        d, _ = self.lstm_e_2(d)
        Q = nn.MaxPool2d(kernel_size=[d.shape[1],1], stride=[1,1])(torch.unsqueeze(d, dim=1))
        Q = torch.squeeze(Q, dim=1)
        K = self.dense_K_e(d)
        V = d
        QK = nn.functional.softmax(torch.sum(torch.multiply(Q, K), dim=-1, keepdims=True) / self.square_d, dim=1)
        elements_2 = torch.sum(torch.multiply(QK, V), dim=1)
        elements_2 = PFILayer(elements_2)
            
        d = torch.nn.functional.embedding(elements[3], self.word_embedding)
        d, _ = self.lstm_e_3(d)
        Q = nn.MaxPool2d(kernel_size=[d.shape[1],1], stride=[1,1])(torch.unsqueeze(d, dim=1))
        Q = torch.squeeze(Q, dim=1)
        K = self.dense_K_e(d)
        V = d
        QK = nn.functional.softmax(torch.sum(torch.multiply(Q, K), dim=-1, keepdims=True) / self.square_d, dim=1)
        elements_3 = torch.sum(torch.multiply(QK, V), dim=1)
        elements_3 = PFILayer(elements_3)
        
        return  [fact_0, fact_1, fact_2, fact_3], [elements_0, elements_1, elements_2, elements_3]




