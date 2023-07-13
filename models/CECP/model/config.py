# -*- coding: utf-8 -*-

class Config():
    def __init__(self):
        
        self.sent_num = 64
        self.sent_len = 32
        self.num_charges = 130  #119/130
        
        self.dropout = 0.5
        self.learning_rate_agent = 0.0001
        self.learning_rate_pred = 0.001
        
        self.epochs = 12
        self.epochs_reinforce = 4 +1   # the last 1 epoch just for evaluating the model
        self.batch_size = 64
        self.hidden_size = 128
        self.word_embedding_size = 200
    
        self.gamma = 0.95               
        self.beta = 0.015  
        self.lambda_2 = 0.10 

        self.display_step = 100
        self.evaluate_train_step = 200
        self.evaluate_test_step = 200
           
        self.seed = 6
        
        
        

