import torch
import torch.nn as nn
import json
import numpy as np

class Mask_Attention(nn.Module):
    def __init__(self):
        super(Mask_Attention, self).__init__()
    def forward(self, query, context):
        attention = torch.bmm(context, query.transpose(1, 2))
        mask = attention.new(attention.size()).zero_()
        mask[:,:,:] = -np.inf
        attention_mask = torch.where(attention==0, mask, attention)
        attention_mask = torch.nn.functional.softmax(attention_mask, dim=-1)
        mask_zero = attention.new(attention.size()).zero_()
        final_attention = torch.where(attention_mask!=attention_mask, mask_zero, attention_mask)
        context_vec = torch.bmm(final_attention, query)
        return context_vec

class Code_Wise_Attention(nn.Module):
    def __init__(self):
        super(Code_Wise_Attention, self).__init__()
    def forward(self,query,context):
        S = torch.bmm(context, query.transpose(1, 2))
        attention = torch.nn.functional.softmax(torch.max(S, 2)[0], dim=-1)
        context_vec = torch.bmm(attention.unsqueeze(1), context)
        return context_vec


# Version of NeurJudge with nn.GRU    
class NeurJudge(nn.Module):
    def __init__(self,embedding,charge_tong_file,art_tong_file,id2charge_file,id2article_file,charge_num,law_num):
        super(NeurJudge, self).__init__()
        self.charge_tong = json.load(open(charge_tong_file))
        self.art_tong = json.load(open(art_tong_file))
        self.id2charge = json.load(open(id2charge_file))
        self.data_size = 200
        self.hidden_dim = 150

        self.embs = nn.Embedding(339503, 200)
        self.embs.weight.data.copy_(embedding)
        self.embs.weight.requires_grad = False

        self.encoder = nn.GRU(self.data_size,self.hidden_dim, batch_first=True, bidirectional=True)
        self.code_wise = Code_Wise_Attention()
        self.mask_attention = Mask_Attention()
        
        self.encoder_term = nn.GRU(self.hidden_dim * 6, self.hidden_dim*3, batch_first=True, bidirectional=True)
        self.encoder_article = nn.GRU(self.hidden_dim * 8, self.hidden_dim*4, batch_first=True, bidirectional=True)

        self.id2article = json.load(open(id2article_file))
        self.mask_attention_article = Mask_Attention()

        self.encoder_charge = nn.GRU(self.data_size,self.hidden_dim, batch_first=True, bidirectional=True)
        self.charge_pred = nn.Linear(self.hidden_dim*6,charge_num)
        self.article_pred = nn.Linear(self.hidden_dim*8,law_num)
        self.time_pred = nn.Linear(self.hidden_dim*6,11)

        self.charge_num=charge_num
        self.law_num=law_num

    def graph_decomposition_operation(self,_label,label_sim,id2label,label2id,num_label,layers=2):
        for _ in range(layers):
            new_label_tong = []
            for index in range(num_label):
                Li = _label[index]
                Lj_list = []
                if len(label_sim[id2label[str(index)]]) == 0:
                    new_label_tong.append(Li.unsqueeze(0))
                else:
                    for sim_label in label_sim[id2label[str(index)]]:
                        Lj = _label[int(label2id[str(sim_label)])]
                        x1 = Li*Lj
                        x1 = torch.sum(x1,-1)
                        x2 = Lj*Lj
                        x2 = torch.sum(x2,-1)
                        x2 = x2+1e-10
                        xx = x1/x2
                        Lj = xx.unsqueeze(-1)*Lj
                        Lj_list.append(Lj)
                    Lj_list = torch.stack(Lj_list,0).squeeze(1)
                    Lj_list = torch.mean(Lj_list,0).unsqueeze(0)
                    new_label_tong.append(Li-Lj_list)  
            new_label_tong = torch.stack(new_label_tong,0).squeeze(1)
            _label = new_label_tong
        return _label

    def fact_separation(self,process,verdict_names,device,embs,encoder,circumstance,mask_attention,types):
        verdict, verdict_len = process.process_law(verdict_names,types)
        verdict = verdict.to(device)
        verdict_len = verdict_len.to(device)
        verdict = embs(verdict)
        verdict_hidden,_ = encoder(verdict)
        # Fact Separation
        scenario = mask_attention(verdict_hidden,circumstance)
        # vector rejection
        x3 = circumstance*scenario
        x3 = torch.sum(x3,2)
        x4 = scenario*scenario
        x4 = torch.sum(x4,2)
        x4 = x4+1e-10
        xx = x3/x4
        # similar vectors
        similar = xx.unsqueeze(-1)*scenario 
        # dissimilar vectors
        dissimilar = circumstance - similar
        return similar,dissimilar
    
    def forward(self,charge,charge_sent_len,article,\
    article_sent_len,charge_tong2id,id2charge_tong,art2id,id2art,\
    documents,sent_lent,process,device):
        # deal the semantics of labels (i.e., charges and articles) 
        charge = self.embs(charge)
        article = self.embs(article)
        charge,_ = self.encoder_charge(charge)
        article,_ = self.encoder_charge(article)
        _charge = charge.mean(1)
        _article = article.mean(1)
        # the original charge and article features
        ori_a = charge.mean(1)
        ori_b = article.mean(1)
        # the number of spreading layers is set as 2
        # GDO for the charge graph
        new_charge = self.graph_decomposition_operation(_label = _charge, label_sim = self.charge_tong, id2label = id2charge_tong, label2id = charge_tong2id, num_label = self.charge_num, layers = 2)

        # GDO for the article graph
        new_article = self.graph_decomposition_operation(_label = _article, label_sim = self.art_tong, id2label = id2art, label2id = art2id, num_label = self.law_num, layers = 2)

        # deal the case fact
        doc = self.embs(documents)
        d_hidden,_ = self.encoder(doc) 
        
        # L2F attention for charges
        new_charge_repeat = new_charge.unsqueeze(0).repeat(d_hidden.size(0),1,1)
        d_hidden_charge = self.code_wise(new_charge_repeat,d_hidden)
        d_hidden_charge = d_hidden_charge.repeat(1,doc.size(1),1)
        
        a_repeat = ori_a.unsqueeze(0).repeat(d_hidden.size(0),1,1)
        d_a = self.code_wise(a_repeat,d_hidden)
        d_a = d_a.repeat(1,doc.size(1),1)

        # the charge prediction
        fact_charge = torch.cat([d_hidden,d_hidden_charge,d_a],-1)
        #fact_charge_hidden = self.l_encoder([fact_charge,sent_lent.view(-1)])
        fact_charge_hidden = fact_charge
        df = fact_charge_hidden.mean(1)
        charge_out = self.charge_pred(df)
        
        charge_pred = charge_out.cpu().argmax(dim=1).numpy()
        charge_names = [self.id2charge[str(i)] for i in charge_pred]
        # Fact Separation for verdicts
        adc_vector, sec_vector = self.fact_separation(process = process,verdict_names = charge_names,device = device ,embs = self.embs,encoder = self.encoder, circumstance = d_hidden, mask_attention = self.mask_attention,types = 'charge')

        # L2F attention for articles
        new_article_repeat = new_article.unsqueeze(0).repeat(d_hidden.size(0),1,1)
        d_hidden_article = self.code_wise(new_article_repeat,d_hidden)
        d_hidden_article = d_hidden_article.repeat(1,doc.size(1),1)

        b_repeat = ori_b.unsqueeze(0).repeat(d_hidden.size(0),1,1)
        d_b = self.code_wise(b_repeat,d_hidden)
        d_b = d_b.repeat(1,doc.size(1),1)
        # the article prediction
        fact_article = torch.cat([d_hidden,d_hidden_article,adc_vector,d_b],-1)
        fact_legal_article_hidden,_ = self.encoder_article(fact_article)

        fact_article_hidden = fact_legal_article_hidden.mean(1)
        article_out = self.article_pred(fact_article_hidden)

        article_pred = article_out.cpu().argmax(dim=1).numpy()
        article_names = [self.id2article[str(i)] for i in article_pred]

        # Fact Separation for sentencing
        ssc_vector, dsc_vector = self.fact_separation(process = process,verdict_names = article_names,device = device ,embs = self.embs,encoder = self.encoder, circumstance = sec_vector, mask_attention = self.mask_attention,types = 'article')

        # the term of penalty prediction change here
        # term_message = torch.cat([ssc_vector,dsc_vector],-1)
        term_message = torch.cat([d_hidden,ssc_vector,dsc_vector],-1)

        term_message,_ = self.encoder_term(term_message)

        fact_legal_time_hidden = term_message.mean(1)
        time_out = self.time_pred(fact_legal_time_hidden)

        return charge_out,article_out,time_out