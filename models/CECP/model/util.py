# -*- coding: utf-8 -*-

import time
import pickle
import numpy as np
import torch
from torch import nn
import multiprocessing
from model import PredNet, ACNet, PredNetPretrain
import copy
from encoder import Encoder


def get_pred_net_pretrain(config):
    pred_net = PredNetPretrain(config)
    return pred_net

def get_pred_net(config):
    pred_net = PredNet(config)
    return pred_net


def get_encoder(config, word_embedding):      
    encoder = Encoder(config, word_embedding)
    return encoder


def get_agent(config):
    agent = ACNet(config=config)
    return agent


def get_pred_logits_one_sample(w, b, context_f, context_e):
    concat_f_e = torch.cat([context_f[0], context_e[0], torch.multiply(context_f[0], context_e[0]),
                            context_f[1], context_e[1], torch.multiply(context_f[1], context_e[1]),
                            context_f[2], context_e[2], torch.multiply(context_f[2], context_e[2]),
                            context_f[3], context_e[3], torch.multiply(context_f[3], context_e[3])], dim=1)
    logits = torch.matmul(concat_f_e, w.T) + b
    return logits


class SharedRMSprop(torch.optim.RMSprop):
    def __init__(self, params, lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0):
        super(SharedRMSprop, self).__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.tensor(0, dtype=torch.int32)
                state['square_avg'] = torch.zeros_like(p.data)

                state['square_avg'].share_memory_()
                state['step'].share_memory_()


def push_and_pull(buffer_rewards, buffer_log_probs, buffer_values, buffer_entropys, optimizer_agent, local_agent, global_agent, 
                  config, lock, global_step, optimizer_local):
    
    buffer_targets = []
    power = 0
    R_ = 0.0
    for i in buffer_rewards[::-1]:     # discount the values in the future
        R_ = i + R_ * config.gamma
        buffer_targets.append(R_)
    buffer_targets.reverse()

    buffer_targets = torch.tensor(buffer_targets)
    
    buffer_log_probs = torch.cat(buffer_log_probs)
    buffer_values = torch.squeeze(torch.cat(buffer_values))
    buffer_entropys = torch.cat(buffer_entropys)
    
    advantage = buffer_targets - buffer_values
    agent_loss = (-buffer_log_probs * advantage.detach() + advantage.pow(2) - config.beta * buffer_entropys).sum()
    optimizer_local.zero_grad()
    agent_loss.backward()

    with lock:
        for local_p, global_p in zip(local_agent.parameters(), global_agent.parameters()):
            global_p._grad = local_p.grad
        optimizer_agent.step()
    
        local_agent.load_state_dict(global_agent.state_dict())
    


# --------------------------------------------------------------------------------------------------------------
def get_batch_iter(epochs, data, config, shuffle=True):
    data = np.array(data)
    size = len(data)
    num_batches_per_peoch = int(size/config.batch_size)
    
    for epoch in range(epochs):
        if shuffle:
            np.random.shuffle(data)
            
        for i in range(num_batches_per_peoch + 1):
            start = i * config.batch_size
            end = min(size, (i+1) * config.batch_size)
            
            percent = i / num_batches_per_peoch
            if end==start:
                continue
            else:
                yield data[start:end], epoch +1, percent


# ---------------------------------------------------------------------------------------------------------------
def get_init_S(batch_x):
    return np.zeros([4, batch_x[0].shape[0], 64])

# ----------------------------------------------------------------------------------------------------------

def get_reward(current_S, current_pred, one_sample_y, action, config):   # current_S, [4, 64]
    current_pred = nn.functional.softmax(current_pred, dim=-1).data.numpy()
    reward = 0.
    
    y_pred = np.argmax(current_pred)
    y_ture = np.argmax(one_sample_y)
    
    if y_pred==y_ture:
        reward += current_pred[0, y_ture]
    else:
        reward += (current_pred[0, y_ture] -1.0)
    
    num_repeat_sentences = sum(current_S[:, action])
    if num_repeat_sentences > 1.0:
        reward -= config.lambda_2 * (num_repeat_sentences-1 / 4.0)
    
    return reward

# --------------------------------------------------------------------------------------------------------------
def load_data(path_data, dataset, scale):
    if dataset=='criminal':
        r = {}
        path_train = path_data + dataset + '_' + scale + '_train.pkl'
        with open(path_train, 'rb') as f:
            data = pickle.load(f)
        r.update({'train':data})

        path_valid = path_data + dataset + '_' + scale + '_valid.pkl'
        with open(path_valid, 'rb') as f:
            data = pickle.load(f)
        r.update({'valid':data})
        
        path_test = path_data + dataset + '_' + scale + '_test.pkl'
        with open(path_test, 'rb') as f:
            data = pickle.load(f)
        r.update({'test':data})
        
        path_elements = path_data + 'elements_criminal.pkl'
        with open(path_elements, 'rb') as f:
            data = pickle.load(f)
        r.update({'elements':data})
        
        return r
    
    if dataset=='cail':
        #我只复现CAIL数据集，所以就只管这个了
        r = {}
        print('开始预处理数据')

        #这一堆是CAIL small的
        for k in ['train','valid','test']:
            path='/data/wanghuijuan/cecp_data/cail_small_'+k+'.pkl'
            with open(path, 'rb') as f:
                data = pickle.load(f)
            r.update({k:data})
            
        path_elements = '/data/wanghuijuan/cecp_data/elements_cail_small.pkl'
        with open(path_elements, 'rb') as f:
            data = pickle.load(f)
        r.update({'elements':data})
        
            
            
        return r
    
    
def get_train_test_val_data(dataset, scale, config):  #这个函数虽然在util.py中写了出来，但是事实上没有用到
    save_encoded_data_path = './encoded_data/data_' + 'LSTM_selfAtt_' + dataset + '_' + scale + '.pkl'
    with open(save_encoded_data_path, 'rb') as f:
        encoded_data = pickle.load(f)
    x_train, y_train = encoded_data['encoded_x_train'], encoded_data['y_train']
    x_test, y_test   = encoded_data['encoded_x_test'],  encoded_data['y_test']
    x_valid, y_valid = encoded_data['encoded_x_valid'], encoded_data['y_valid']
    assert y_train.shape[1]==y_test.shape[1]==y_valid.shape[1]==config.num_charges
        
    elements = encoded_data['encoded_elements']
    similarity_weights = encoded_data['similarity_weights']
    pred_weights = encoded_data['pred_weights']
    pred_bias = encoded_data['pred_bias']
    
    # ----------------------------------------------------
    path_data = './data/data/'
    data = load_data(path_data, dataset, scale)
    
    sent_num_train = data['train']['sent_num']
    sent_num_test  = data['test']['sent_num']
    sent_num_valid = data['valid']['sent_num']
    
    # ---------------------------------------------------------
    # Convert to torch tensor
    elements[0] = torch.from_numpy(elements[0])
    elements[1] = torch.from_numpy(elements[1])
    elements[2] = torch.from_numpy(elements[2])
    elements[3] = torch.from_numpy(elements[3])
    
    if config.debug < 0:
        return elements, [similarity_weights, pred_weights, pred_bias], \
            x_train, y_train, sent_num_train, x_test, y_test, sent_num_test, x_valid, y_valid, sent_num_valid
    else:
        return elements, [similarity_weights, pred_weights, pred_bias], \
            x_train[:128], y_train[:128], sent_num_train[:128], x_test, y_test, sent_num_test, x_valid, y_valid, sent_num_valid



# ------------------------------------------------------------------------------------------------------------------------
def get_date_time():
    w_time = time.localtime(time.time())
    w_time = [str(w_time.tm_year), str(w_time.tm_mon), str(w_time.tm_mday), '  ', str(w_time.tm_hour), str(w_time.tm_min)]
    w_time = ':'.join(w_time)
    
    return w_time


# ------------------------------------------------------------------------------------------------------------------------
def get_train_parameters(args, config):

    dataset = args.dataset
    scale = args.scale
    num_workers = multiprocessing.cpu_count()

    if args.bs > -1:
        config.batch_size = args.bs
    if args.nclass > -1:
        config.num_charges = args.nclass
    if args.lr > -1:
        config.learning_rate_agent = args.lr
    if args.cpun > -1:
        num_workers = args.cpun
    if args.beta > -1:
        config.beta = args.beta
    if args.lrp > -1:
        config.learning_rate_pred = args.lrp

    path_log_train = '../logs/' + dataset + '_' + scale + '_train_log'
    path_log_train_eval = '../logs/' + dataset + '_' + scale + '_evaluation_train_log'
    path_log_test_eval = '../logs/' + dataset + '_' + scale + '_evaluation_test_log'
    path_log_val_eval = '../logs/' + dataset + '_' + scale + '_evaluation_val_log'
    
    ckpt_path='/data/wanghuijuan/cecp_data/ckpt/'
    path_save_models = {}
    path_save_models['encoder'] = ckpt_path + dataset + '_' + scale + '_encoder_epoch_'
    path_save_models['prednet'] = ckpt_path + dataset + '_' + scale + '_prednet_epoch_'
    path_save_models['glagent'] = ckpt_path + dataset + '_' + scale + '_glagent_epoch_'
    path_save_models['optenpr'] = ckpt_path + dataset + '_' + scale + '_optenpr_epoch_'
    path_save_models['optglag'] = ckpt_path + dataset + '_' + scale + '_optglag_epoch_'

    
    return config, num_workers, path_log_train, path_log_train_eval, path_log_test_eval, path_log_val_eval, path_save_models
        


def load_word_embedding_data(path_word_embedding_data):
    with open(path_word_embedding_data, 'rb') as f:
        data = pickle.load(f)
        
    word_embedding = data['word_embedding']
    word2id = data['word2id']
    id2word = data['id2word']
    assert word_embedding.shape[0]==len(word2id)
    return word_embedding, word2id, id2word





