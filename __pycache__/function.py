# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 23:22:56 2020

@author: 龙
"""

import torch
from torch.utils.data import Dataset
#from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from torch.autograd import Variable

#数据上采样的包
#from collections import Counter#查看数据Ylabel分布
#from sklearn.datasets import make_classification 
from imblearn.over_sampling import SMOTE 
# 定义几个全局变量
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN

# In[45]:




# In[46]:


class DiabetesDataset(Dataset):
    def __init__(self,filepath1,filepath2,filepath3):
        
        list1 = np.load(filepath1)
        list2 = np.load(filepath2)
        list3 = np.concatenate([list1,list2],axis=1)
        data_y = pd.read_csv(filepath3,sep='\t')
        y_label = data_y['Label']
        y_label = np.array(y_label)
        y_label = y_label.reshape(len(y_label),1)
        
        ######制作one_hot
        
#        y_label= np.array([[1 if i==x else 0 for i in range(2)] for x in
#                            y_label]).reshape(-1,1)
        
        self.len = list3.shape[0]
        self.x_data = torch.from_numpy(list3)
        self.y_data = torch.from_numpy(y_label.astype(np.float32)).long()
        
    def __getitem__(self,index):
        
        return self.x_data[index] , self.y_data[index]
    
    def __len__(self):
        return self.len

    

# =============================================================================



#########################
        
  #  进行数据上采样数据读取函数
    
#########################
class DiabetesDataset1(Dataset):
    def __init__(self,filepath1,filepath2,filepath3):
        
        list1 = np.load(filepath1)
        list2 = np.load(filepath2)
        list3 = np.concatenate([list1,list2],axis=1)
        x_data = list3
        data_y = pd.read_csv(filepath3,sep='\t')
        y_label = data_y['Label']
        y_label = np.array(y_label)
        y_label = y_label.reshape(len(y_label),1)
        
        ######进行上采样
        sm = SMOTE(random_state=0)    # 处理过采样的方法
        x_train_sm,y_train_sm = sm.fit_sample(x_data,y_label)

        y_train_sm=y_train_sm.reshape(-1,1)
        
#        y_label= np.array([[1 if i==x else 0 for i in range(2)] for x in
#                            y_label]).reshape(-1,1)
        
        self.len = x_train_sm.shape[0]
        self.x_data = torch.from_numpy(x_train_sm)
        self.y_data = torch.from_numpy(y_train_sm.astype(np.float32)).long()
        
    def __getitem__(self,index):
        
        return self.x_data[index] , self.y_data[index]
    
    def __len__(self):
        return self.len
    
##########################下采样实现
class DiabetesDataset2(Dataset):
    def __init__(self,filepath1,filepath2,filepath3):
        
        list1 = np.load(filepath1)
        list2 = np.load(filepath2)
        list3 = np.concatenate([list1,list2],axis=1)
        x_data = list3
        data_y = pd.read_csv(filepath3,sep='\t')
        y_label = data_y['Label']
        y_label = np.array(y_label)
        y_label = y_label.reshape(len(y_label),1)
        
        ######进行下采样
        model_RandomUnderSampler = RandomUnderSampler()   # 处理采样的方法
        x_train_sm,y_train_sm =model_RandomUnderSampler.fit_sample(x_data,y_label)

        y_train_sm=y_train_sm.reshape(-1,1)
        
#        y_label= np.array([[1 if i==x else 0 for i in range(2)] for x in
#                            y_label]).reshape(-1,1)
        
        self.len = x_train_sm.shape[0]
        self.x_data = torch.from_numpy(x_train_sm)
        self.y_data = torch.from_numpy(y_train_sm.astype(np.float32)).long()
        
    def __getitem__(self,index):
        
        return self.x_data[index] , self.y_data[index]
    
    def __len__(self):
        return self.len

class DiabetesDataset3(Dataset):
    def __init__(self,filepath1,filepath2,filepath3):
        
        list1 = np.load(filepath1)
        list2 = np.load(filepath2)
        list3 = np.concatenate([list1,list2],axis=1)
        x_data = list3
        data_y = pd.read_csv(filepath3,sep='\t')
        y_label = data_y['Label']
        y_label = np.array(y_label)
        y_label = y_label.reshape(len(y_label),1)
        
        ######进行上采样
        sm = ADASYN(random_state = 0)    # 处理过采样的方法
        x_train_sm,y_train_sm = sm.fit_sample(x_data,y_label)

        y_train_sm=y_train_sm.reshape(-1,1)
        
#        y_label= np.array([[1 if i==x else 0 for i in range(2)] for x in
#                            y_label]).reshape(-1,1)
        
        self.len = x_train_sm.shape[0]
        self.x_data = torch.from_numpy(x_train_sm)
        self.y_data = torch.from_numpy(y_train_sm.astype(np.float32)).long()
        
    def __getitem__(self,index):
        
        return self.x_data[index] , self.y_data[index]
    
    def __len__(self):
        return self.len  
    
class LINEAR(torch.nn.Module):
    def __init__(self):
        super(LINEAR,self).__init__()
        self.linear1 = torch.nn.Linear(1536,768)
        self.linear2 = torch.nn.Linear(768,256)
        self.linear3 = torch.nn.Linear(256,64)
        self.linear4 = torch.nn.Linear(64,16)
        self.linear5 = torch.nn.Linear(16,4)
        self.linear6 = torch.nn.Linear(4,2)
        #self.sigmoid = torch.nn.ReLU()#不同的激活函数的尝试
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self,x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        x = self.sigmoid(self.linear5(x))
        x = self.sigmoid(self.linear6(x))
        return x    
    
    
    
    
    
    
    
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
class GRU(torch.nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(GRU,self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.gru=torch.nn.GRU(input_size,hidden_size,num_layers,batch_first=True)
        self.fc=torch.nn.Linear(hidden_size,num_classes)
        self.sigmoid=torch.nn.Sigmoid()
        
    def forward(self,x):
        h0=Variable(torch.zeros(self.num_layers,x.size(0),self.hidden_size))
        h0=h0.to(device)
        o,_=self.gru(x,h0)
        
        o=o[:,-1,:]
        o=self.fc(o)
        o=self.sigmoid(o)
        return o
        
#=========================定义训练函数，以备调用======================
class BiLSTM(torch.nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(BiLSTM,self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm=torch.nn.LSTM(input_size,hidden_size,num_layers,
                                batch_first=True,bidirectional=True)
        self.fc1=torch.nn.Linear(hidden_size*2,hidden_size)
        self.fc2=torch.nn.Linear(hidden_size,4)
        self.fc3=torch.nn.Linear(4,num_classes)
        self.relu=torch.nn.ReLU()
        self.softmax=torch.nn.Softmax()
    def forward(self,x):
        h0=torch.randn(self.num_layers*2,x.size(0),self.hidden_size)
        h0=h0.to(device)
        c0=torch.randn(self.num_layers*2,x.size(0),self.hidden_size)
        c0=c0.to(device)
        out,_=self.lstm(x,(h0,c0))
        out= out[:,-1,:]
        out=self.fc1(out)
        out=self.relu(out)
        out=self.fc2(out)
        out=self.softmax(self.relu(out))
        
        out=self.fc3(out)
        out=self.softmax(self.relu(out))
        #out=self.softmax(out)
        
        return out
    
class LSTM(torch.nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(LSTM,self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm=torch.nn.LSTM(input_size,hidden_size,num_layers,
                                batch_first=True,bidirectional=False)
        self.fc=torch.nn.Linear(hidden_size*1,num_classes)
        self.sigmoid=torch.nn.Sigmoid()
        self.softmax=torch.nn.Softmax()
    def forward(self,x):
        h0=torch.randn(self.num_layers*1,x.size(0),self.hidden_size)
        h0=h0.to(device)
        c0=torch.randn(self.num_layers*1,x.size(0),self.hidden_size)
        c0=c0.to(device)
        out,_=self.lstm(x,(h0,c0))
        out=self.fc(out[:,-1,:])
        
        return out
        
