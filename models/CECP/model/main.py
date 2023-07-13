# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(6)
import torch
torch.manual_seed(6)
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import multiprocessing
import time
import argparse
from sklearn.preprocessing import OneHotEncoder
import torch.nn as nn
import pickle
from tqdm import tqdm

from config import Config
from model import Environment
from util import get_train_test_val_data, get_train_parameters, get_batch_iter, get_reward, get_init_S, load_data, load_word_embedding_data
from util import get_pred_net, get_agent, get_pred_logits_one_sample, SharedRMSprop, push_and_pull, get_encoder, get_pred_net_pretrain

import warnings
import os
import copy
import sys
warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["OMP_NUM_THREADS"] = "1"    # prevent numpy from using multiple threads


# ----------------------------------------------------------------------------------------------------------------------------------------
def Worker(process_id, global_agent, optimizer_agent, config, asyn_data, batch_S, batch_rewards, batch_index, batch_grads,
                                           current_train_nums, lock, global_step, asyn_flag):
    env = Environment()
    local_agent = get_agent(config)
    optimizer_local = SharedRMSprop(params=local_agent.parameters(), lr=config.learning_rate_agent)
    
    while True:
        if asyn_flag['flag_process']:
            local_agent.load_state_dict(global_agent.state_dict())
            
            while batch_index.value < current_train_nums.value:
                with lock:
                    current_index = batch_index.value
                    batch_index.value += 1
                if current_index > (current_train_nums.value-1):
                    asyn_flag['flag_process'] = False
                    
                    break
        
                elements_encoded = asyn_data['e']
                one_sample_x = [asyn_data['x'][0][current_index], asyn_data['x'][1][current_index],
                                asyn_data['x'][2][current_index], asyn_data['x'][3][current_index]]
                one_sample_y =  asyn_data['y'][current_index]
                one_sample_sent_num = asyn_data['num'][current_index]
                one_sample_init_S = asyn_data['init_S'][:, current_index, :]     
            
                buffer_rewards, buffer_values, buffer_log_probs, buffer_entropys = [], [], [], []
                return_rewards = []
                p = 0
                step = 0
                selected_sente = env.reset(fact=one_sample_x, S=one_sample_init_S.copy(), elements=elements_encoded, 
                                           aggregate_weights=asyn_data['prednet_aggregation']).copy()
                local_agent.customer_init(selected_sente, elements_encoded)
        
                step_flags, step_records = [6, 3, 3, 6], [0, 0, 0, 0]
                while step_records[0]<step_flags[0] or step_records[1]<step_flags[1] or step_records[2]<step_flags[2] \
                    or step_records[3]<step_flags[3]:
                    
                    if step_records[p] >= step_flags[p]:
                        p += 1
                        if p==4:
                            p = 0
                        continue
                    step_records[p] += 1
                    step += 1

                    S_row = env.S[p].reshape([-1, 1]).copy()
                    action_prob, value = local_agent(selected_sente[p], one_sample_x[p], elements_encoded[p], p, one_sample_sent_num, S_row)
                    distribution = torch.distributions.Categorical(probs=action_prob)
                        
                    if asyn_flag['training']:
                        action = distribution.sample()
                        log_prob = distribution.log_prob(action)
                        entropy = distribution.entropy()
                        action = action.numpy()[0]
                    else:
                        action_p, action = action_prob.max(dim=1)
                        action = action.numpy()[0]
                        
                    if True:
                        selected_sente_ = env.step(action, p, elements_encoded[p], asyn_data['prednet_aggregation']).copy()
                        current_S, current_f, current_e = env.S.copy(), env.context_f, env.context_e
                        
                        current_pred_logits = get_pred_logits_one_sample(asyn_data['pred_w'], asyn_data['pred_b'], current_f, current_e)
                        reward = get_reward(current_S, current_pred_logits, one_sample_y, action, config)

                        buffer_rewards.append(reward)
                        return_rewards.append(reward)
                        buffer_values.append(value)
                        if asyn_flag['training']:
                            buffer_log_probs.append(log_prob)
                            buffer_entropys.append(entropy)

                    selected_sente = selected_sente_
        
                    p += 1
                    if p==4:
                        p = 0
                

                if asyn_flag['training']:
                    push_and_pull(buffer_rewards, buffer_log_probs, buffer_values, buffer_entropys, 
                                  optimizer_agent, local_agent, global_agent, 
                                  config, lock, global_step, optimizer_local)
                    buffer_rewards, buffer_values, buffer_log_probs, buffer_entropys = [], [], [], []
                    
                with lock:
                    batch_rewards.update({current_index:return_rewards[-1]})
                    batch_S.update({current_index:current_S})    
            
            asyn_flag['flag_process'] = False

# --------------------------------------------------------------------------------------------------------------------------

def main():
    # -----------------------------------------------------------------------------------------------------------
    ''' parameter initialization '''
    parser = argparse.ArgumentParser(description='None')
    parser.add_argument('--dataset', type=str, default='cail')
    parser.add_argument('--scale', type=str, default='big')  #small/big
    parser.add_argument('--nclass', type=int, default=130)  #small 119, big 130
    parser.add_argument('--eval', type=int, default=1)
    
    parser.add_argument('--bs', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=-1.0)
    parser.add_argument('--cpun', type=int, default=16)
    parser.add_argument('--beta', type=float, default=-1)
    parser.add_argument('--lrp', type=float, default=-1.0)
    parser.add_argument('--gpu', type=int, default=0)
       
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    config = Config()
    global_step, best_f1_val = 0, 0
    config, num_workers, path_log_train, path_log_train_eval, path_log_test_eval, path_log_val_eval, path_save_models \
                    = get_train_parameters(args, config)
    
    path_data = '../data/processed_data/'
    
    #small数据
    #重写数据加载部分

    with open('/home/wanghuijuan/whj_files/github_projects/LADAN/data_and_config/data/w2id_thulac.pkl', 'rb') as f:
        word2id_dict = pickle.load(f)
    embedding_file_path='/home/wanghuijuan/whj_files/cail_help/ladan_files/cail_thulac.npy'
    embedding_weight=np.load(embedding_file_path)
    embedding_dim=embedding_weight.shape[1]  #200，与CECP的相同
    embedding_weight[word2id_dict['BLANK']]=[0 for _ in range(embedding_dim)]
    
    #重写数据加载部分
    data = load_data(path_data, args.dataset, args.scale)
    
    #重写部分结束
    word_embedding = embedding_weight.astype(np.float32)
    
    x_train, y_train = data['train']['x'], data['train']['y']
    sent_num_train = data['train']['sent_num']
    x_test, y_test = data['test']['x'], data['test']['y']
    sent_num_test  = data['test']['sent_num']

    x_valid, y_valid = data['valid']['x'],  data['valid']['y']
    sent_num_valid = data['valid']['sent_num']
        
    elem_subject, elem_subjective, elem_object, elem_objective = data['elements']['ele_subject'], \
                data['elements']['ele_subjective'], data['elements']['ele_object'], data['elements']['ele_objective']
    
    elements_ori = [torch.tensor(elem_objective, dtype=torch.long), torch.tensor(elem_subject, dtype=torch.long), 
                    torch.tensor(elem_subjective, dtype=torch.long), torch.tensor(elem_object, dtype=torch.long)]
    elements_ori = [e.cuda() for e in elements_ori]
    
    one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.float32)
    y_train = one_hot_encoder.fit_transform(np.array(y_train).reshape(len(y_train), 1))
    y_test = one_hot_encoder.transform(np.array(y_test).reshape(len(y_test), 1))
    y_valid = one_hot_encoder.transform(np.array(y_valid).reshape(len(y_valid), 1))
    assert y_train.shape[1]==y_test.shape[1]==y_valid.shape[1]==config.num_charges
    
    print('train nums:', y_train.shape[0], ' test nums:', y_test.shape[0], ' class nums:', y_train.shape[1])
    
    
    # ------------------------------------------------------------------------------------------------------------
    ''' pretrain '''
    encoder_pretrain = get_encoder(config, word_embedding).cuda()
    pred_net_pretrain = get_pred_net_pretrain(config).cuda()
    encoder_pred_net_pretrain = torch.nn.Sequential(encoder_pretrain, pred_net_pretrain)
    optimizer = torch.optim.Adam(params=encoder_pred_net_pretrain.parameters(), lr=config.learning_rate_pred)
    
    batches_pretrain = get_batch_iter(config.epochs, data=list(zip(x_train, y_train)),  config=config, shuffle=True)
    global_step = 0
    encoder_pred_net_pretrain.train()
    for batch in batches_pretrain:
        t0 = time.time()
        global_step += 1
        data, epoch, percent = batch
        
        batch_x, batch_y = zip(*data)
        batch_x, batch_y = torch.tensor(batch_x, dtype=torch.long).cuda(), torch.tensor(batch_y)
        
        a, b = encoder_pretrain(batch_x, elements_ori)
        batch_pred_logits = pred_net_pretrain(a, b)
        _, target_y = batch_y.cuda().max(dim=1)
        batch_pred_loss = torch.nn.functional.cross_entropy(input=batch_pred_logits, target=target_y, reduction='mean')
        loss = batch_pred_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                
        batch_pred_labels = np.argmax(batch_pred_logits.cpu().detach().numpy(), axis=1)
            
        acc = accuracy_score(y_pred=batch_pred_labels, y_true=np.argmax(batch_y, axis=1))
        precision = precision_score(average='macro', y_pred=batch_pred_labels, y_true=np.argmax(batch_y, axis=1))
        recall = recall_score(average='macro', y_pred=batch_pred_labels, y_true=np.argmax(batch_y, axis=1))
        f1 = f1_score(average='macro', y_pred=batch_pred_labels, y_true=np.argmax(batch_y, axis=1))
        
        display = 'epoch=%d, step=%d, percent=%.1f%%, loss_pre=%.2f, \
                    acc=%.4f, f1=%.4f, precision=%.4f, recall=%.4f, use_time=%.2f' % \
                (epoch, global_step, percent*100, 
                 batch_pred_loss.cpu().detach().numpy(),
                 acc, f1, precision, recall, time.time() - t0)
    
        if args.scale=='big':
            config.display_step = 1000
        if global_step % config.display_step == 0:
            print(display)
    if args.scale=='big':
        torch.save(encoder_pretrain.state_dict(), path_save_models['encoder'] + 'epoch_12_' + \
                   time.asctime(time.localtime(time.time())).replace(' ', '_').replace(':', '_')[4:] + '.pt')
        torch.save(pred_net_pretrain.state_dict(), path_save_models['prednet'] + 'epoch_12_' + \
                   time.asctime(time.localtime(time.time())).replace(' ', '_').replace(':', '_')[4:] + '.pt')
        
            
    # -------------------------------------------------------------------------------------------------------    
    ''' train setting'''
    encoder = encoder_pretrain
    pred_net = get_pred_net(config).cuda()
    pred_net.load_state_dict(pred_net_pretrain.state_dict())
    
    global_agent = get_agent(config)
    encoder_pred_net = torch.nn.Sequential(encoder, pred_net)
        
    # parameter
    lock = multiprocessing.Lock()
    manager = multiprocessing.Manager()
    batch_index = multiprocessing.Value('i', 0)
    current_train_nums = multiprocessing.Value('i', 0)
    batch_S, asyn_data, batch_rewards, batch_grads, asyn_flag = \
        manager.dict(), manager.dict(), manager.dict(), manager.dict(), manager.dict()
    
    
    batches_train = get_batch_iter(config.epochs_reinforce, data=list(zip(x_train, y_train, sent_num_train)), config=config, shuffle=True)
    optimizer_pre = torch.optim.Adam(params=encoder_pred_net.parameters(), lr=config.learning_rate_pred)
    optimizer_agent = SharedRMSprop(params=global_agent.parameters(), lr=config.learning_rate_agent)
    
    # -----------------------------------------------------------------------------------------------------------------------
    worker_processes = []
    asyn_flag['flag_process'] = False
    asyn_flag['training'] = True
    
    for process_id in range(num_workers):
        worker_processes.append(multiprocessing.Process(target=Worker, 
                                                        args=(process_id, global_agent, optimizer_agent, 
                                                              config, asyn_data, 
                                                              batch_S, batch_rewards, batch_index, batch_grads, 
                                                              current_train_nums, lock, global_step, asyn_flag)))
    for p in worker_processes:
        p.start()

#-----------------------------------------------------------------------------------------------------------------------------------    
    def evaluate_test_val(x, y, n, flg, global_step, g_epoch, pred_net, encoder):
        if flg=='test':
            print('Evaluate Test Set...... ', end='')
        if flg=='val':
            print('Evaluate Val Set...... ', end='')
        asyn_data['flag_process'] = False
        
        rewards_eval, loss_eval, pred_eval = [], [], []
        eval_batches = get_batch_iter(1, data=list(zip(x, y, n)), config=config, shuffle=False)
        y_eval_vec = np.argmax(y, axis=1)
        t0 = time.time()
        
        
        for eval_batch in eval_batches:
            data, epoch, percent = eval_batch
            batch_x, batch_y, batch_sent_num = zip(*data)
            batch_x, batch_y = torch.tensor(batch_x, dtype=torch.long), torch.tensor(batch_y)
            batch_x, elements = encoder(batch_x.cuda(), elements_ori)
            
            batch_S.clear()
            batch_rewards.clear()
            batch_grads.clear()
            asyn_data.clear()
            batch_index.value = 0
            current_train_nums.value = batch_x[0].shape[0]
            asyn_data['e'] = [e.cpu().detach() for e in elements]
            asyn_data['x'] = [x.cpu().detach() for x in batch_x]
            asyn_data['y'] = batch_y
            asyn_data['num'] = batch_sent_num
            asyn_data['init_S'] = get_init_S([x.cpu().detach() for x in batch_x])
            asyn_data['prednet_aggregation'] = [pred_net.state_dict()['aggregation_layer_0.weights'].cpu().detach(),
                                               pred_net.state_dict()['aggregation_layer_1.weights'].cpu().detach(),
                                               pred_net.state_dict()['aggregation_layer_2.weights'].cpu().detach(),
                                               pred_net.state_dict()['aggregation_layer_3.weights'].cpu().detach()]    
            asyn_data['pred_w'] = pred_net.state_dict()['dense_pred.weight'].cpu().detach()
            asyn_data['pred_b'] = pred_net.state_dict()['dense_pred.bias'].cpu().detach()
            
            asyn_flag['training'] = False
            asyn_flag['flag_process'] = True
            
            while True:
                if len(np.unique(list(batch_S.keys()))) == current_train_nums.value:
                    asyn_flag['flag_process'] = False
                    
                    batch_pred_logits = pred_net(batch_x, elements, batch_S)
                    _, target_y = batch_y.cuda().max(dim=1)
                    loss = torch.nn.functional.cross_entropy(input=batch_pred_logits, target=target_y, reduction='mean')
                    
                    batch_pred_labels = np.argmax(batch_pred_logits.cpu().detach().numpy(), axis=1)
                    pred_eval.extend(batch_pred_labels)
                    loss_eval.append(loss.cpu().detach().numpy())
                    rewards_eval.extend(list(batch_rewards.values())[:])
                    
                    break

        print('Total Number:', len(pred_eval)) 
        assert len(pred_eval)==x.shape[0]
        loss_eval = np.mean(loss_eval)
        acc_eval = accuracy_score(y_pred=pred_eval, y_true=y_eval_vec)
        f1_eval = f1_score(y_pred=pred_eval, y_true=y_eval_vec, average='macro')
        precision_eval = precision_score(y_pred=pred_eval, y_true=y_eval_vec, average='macro')
        recall_eval = recall_score(y_pred=pred_eval, y_true=y_eval_vec, average='macro')
            
        if flg=='test':
            display = '   Test: step=%d, epoch=%d, loss=%.4f, acc=%.4f, f1=%.4f, precision=%.4f, recall=%.4f, use_time=%.2fmin' % \
                (global_step, g_epoch-1, loss_eval,
                 acc_eval, f1_eval, precision_eval, recall_eval, (time.time() - t0)/60.0)
            print(display)
            
        if flg=='val':
            display = '   Val: step=%d, epoch=%d, loss=%.4f, acc=%.4f, f1=%.4f, precision=%.4f, recall=%.4f, use_time=%.2fmin' % \
                (global_step, g_epoch-1, loss_eval, 
                 acc_eval, f1_eval, precision_eval, recall_eval, (time.time() - t0)/60.0)
            print(display)
            
        return f1_eval, display
    
    
# -----------------------------------------------------------------------------------------------------------------------------------
    try: 
        print(config.__dict__)
        print('start training ...')
        
        eval_train_y, eval_train_pred = [], []
        for batch in batches_train:
            global_step += 1
            t0 = time.time()
            data, epoch, percent = batch
            
            if args.eval > -1 and epoch > 1 and percent==0:
                
                print()
                encoder.eval()
                pred_net.eval()
                encoder_pred_net.eval()

                f_v, display_v = evaluate_test_val(x_valid, y_valid, sent_num_valid, 'val', global_step, epoch, pred_net, encoder)
                f_t, display_t = evaluate_test_val(x_test, y_test, sent_num_test, 'test', global_step, epoch, pred_net, encoder)

                if f_v > best_f1_val:
                    best_f1_val = f_v
                    best_f1_test = f_t
                    print('Update f1', best_f1_val, best_f1_test)
                    print('Train f1', f1_score(y_pred=eval_train_pred, y_true=eval_train_y, average='macro'), '\n')
                    display = 'Train: epoch=%d, acc=%.4f, f1=%.4f' % \
                        (epoch-1, accuracy_score(y_pred=eval_train_pred, y_true=eval_train_y), 
                         f1_score(y_pred=eval_train_pred, y_true=eval_train_y, average='macro'))
                    
                    eval_train_y, eval_train_pred = [], []
                else:
                    print('No Improvement', best_f1_val, best_f1_test)
                    print('Train f1', f1_score(y_pred=eval_train_pred, y_true=eval_train_y, average='macro'), '\n')
                    display = 'Train: epoch=%d, acc=%.4f, f1=%.4f' % \
                        (epoch-1, accuracy_score(y_pred=eval_train_pred, y_true=eval_train_y), 
                         f1_score(y_pred=eval_train_pred, y_true=eval_train_y, average='macro'))
                    
                    eval_train_y, eval_train_pred = [], []
                
                if epoch == config.epochs_reinforce:   # iterator returns epoch +1, when epoch=6, the model has trained 4 epoches actually
                    with open('../logs/' + args.dataset + '_' + args.scale, 'a+', encoding='utf-8') as f:
                        f.write(time.asctime(time.localtime(time.time())) + '\n' + display_v[3:] + '\n' + display_t[3:] + '\n')
                    break
            
            # ===========================================================================================================
            try:
                batch_x, batch_y, batch_sent_num = zip(*data)
            except:
                print(data)
                raise Exception()
            batch_x, batch_y = torch.tensor(batch_x, dtype=torch.long), torch.tensor(batch_y)

            encoder.train()
            encoder_pred_net.train()
            batch_x, elements = encoder(batch_x.cuda(), elements_ori)
            
            batch_S.clear()
            batch_rewards.clear()
            batch_grads.clear()
            asyn_data.clear()
            batch_index.value = 0
            current_train_nums.value = batch_x[0].shape[0]
            asyn_data['e'] = [e.cpu().detach() for e in elements]
            asyn_data['x'] = [x.cpu().detach() for x in batch_x]
            asyn_data['y'] = batch_y
            asyn_data['num'] = batch_sent_num

            asyn_data['init_S'] = get_init_S([x.cpu().detach() for x in batch_x])
            
            asyn_data['prednet_aggregation'] = [pred_net.state_dict()['aggregation_layer_0.weights'].cpu().detach(),
                                               pred_net.state_dict()['aggregation_layer_1.weights'].cpu().detach(),
                                               pred_net.state_dict()['aggregation_layer_2.weights'].cpu().detach(),
                                               pred_net.state_dict()['aggregation_layer_3.weights'].cpu().detach()]  
            asyn_data['pred_w'] = pred_net.state_dict()['dense_pred.weight'].cpu().detach()
            asyn_data['pred_b'] = pred_net.state_dict()['dense_pred.bias'].cpu().detach()
            
            asyn_flag['training'] = True
            asyn_flag['flag_process'] = True
            
            while True:
                if len(np.unique(list(batch_S.keys()))) == current_train_nums.value:
                    pred_net.train()
                    asyn_flag['flag_process'] = False
                    
                    batch_pred_logits = pred_net(batch_x, elements, batch_S)

                    _, target_y = batch_y.cuda().max(dim=1)
                    batch_pred_loss = torch.nn.functional.cross_entropy(input=batch_pred_logits, target=target_y, reduction='mean')
                        
                    loss = batch_pred_loss
                    l_e = torch.tensor(0.)
                        
                    optimizer_pre.zero_grad()
                    loss.backward()
                    optimizer_pre.step()
                
                    batch_pred_labels = np.argmax(batch_pred_logits.cpu().detach().numpy(), axis=1)
                    acc = accuracy_score(y_pred=batch_pred_labels, y_true=np.argmax(batch_y, axis=1))
                    precision = precision_score(average='macro', y_pred=batch_pred_labels, y_true=np.argmax(batch_y, axis=1))
                    recall = recall_score(average='macro', y_pred=batch_pred_labels, y_true=np.argmax(batch_y, axis=1))
                    f1 = f1_score(average='macro', y_pred=batch_pred_labels, y_true=np.argmax(batch_y, axis=1))
                
                    display = 'epoch=%d, step=%d, p=%.4f%%, loss_pre=%.4f, loss_ele=%.2f, rewards=%.2f, ' % \
                                (epoch, global_step, percent*100, batch_pred_loss.item(), l_e.item(), np.mean(list(batch_rewards.values())))
                    display += 'acc=%.4f, f1=%.4f, precision=%.4f, recall=%.4f, use_time=%.2f' % \
                                (acc, f1, precision, recall, time.time() - t0)
                                
                    # if global_step % config.display_step == 0:
                    if global_step % 1 == 0:
                        # print(display)
                        print('\r', end='')
                        sys.stdout.write(display)
                        sys.stdout.flush()
                    
                    
                    eval_train_y.extend(list(np.argmax(batch_y, axis=1)))
                    eval_train_pred.extend(list(batch_pred_labels))
                    
                    break
               
                
# -------------------------------------------------------------------------------------------------------------------------------
    except KeyboardInterrupt:
        pass


# --------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()

        
