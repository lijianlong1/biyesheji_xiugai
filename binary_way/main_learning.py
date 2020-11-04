# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:45:07 2020

@author: 龙
"""



import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import function as func
from torch.autograd import Variable
import torch.nn as nn
# 定义几个全局变量

# In[45]:


train_loss=[]
test_loss=[]
train_loss_epoch=[]
test_loss_epoch=[]


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

    
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
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
# =============================================================================
class GRU(torch.nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(GRU,self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.gru=torch.nn.GRU(input_size,hidden_size,num_layers,batch_first=True)
        self.fc=torch.nn.Linear(hidden_size,num_classes)
        
        
    def forward(self,x):
        h0=Variable(torch.zeros(self.num_layers,x.size(0),self.hidden_size))
        h0=h0.to(device)
        o,_=self.gru(x,h0)
        
        o=o[:,-1,:]
        o=self.fc(o)
        return o
        
#=========================定义训练函数，以备调用======================
def train(epoch):
    train_loss_sum=0.0
    
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target=data
        optimizer.zero_grad()
        
        #######################################适合RNN的shape
#        inputs=Variable(inputs.view(-1,seq_len,input_size))
#        target=Variable(target)
        ################################################    
        #加上gpu执行语句
        
        inputs,target=inputs.to(device),target.to(device)
        
        target=target.long().squeeze()
        
        outputs=model(inputs)
        
        loss=criterion(outputs,target)
        loss.backward()
        optimizer.step()
        #running_loss+=loss.item()
        train_loss.append(loss.item())
#        if (batch_idx+1)%10==0:
#            print('[%d,%5d]loss:%.3f'%(epoch+1,batch_idx+1,running_loss))
#            running_loss=0.0
    for i in train_loss:
        train_loss_sum+=i
    
    
    train_loss_ave=train_loss_sum/len(train_loss)
    train_loss_epoch.append(train_loss_ave)
                    

                        

                           
    
def test():
#    correct_1=0
#    total_1=0
    acc=0.0
    test_loss_sum=0.0
    TP=0.0
    TN=0.0
    FP=0.0
    FN=0.0
    F1=0.0
    acc=0.0
    with torch.no_grad():
        for data in test_loader:
            images,labels=data
            
            ######将数据转换成适合RNN的shape#####
#            images=Variable(images.view(-1,seq_len,input_size))
#            labels=Variable(labels)
            
            ######################################
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            
            
            _,predicted=outputs.data.max(dim=1)
            labels1=labels.squeeze()
            loss=criterion(outputs,labels1)
            
            test_loss.append(loss.item())
            #labels=torch.topk(labels,1)[1].squeeze(1)
            print('++++++++++++++++++++++++',epoch+1,'++++++++++++++++++++++++')
            prediction = predicted.cpu().numpy()
            print(np.all(prediction.reshape(-1,1)==0))
            # TP predict 和 label 同时为1
            TP += float(((predicted == 1) & (labels1.data == 1)).cpu().sum())
# TN predict 和 label 同时为0
            TN += float(((predicted == 0) & (labels1.data == 0)).cpu().sum())
            FN += float(((predicted == 0) & (labels1.data == 1)).cpu().sum())
# FP predict 1 label 0)
            FP += float(((predicted == 1) & (labels1.data == 0)).cpu().sum())
            
            print(TP,TN,FN,FP)
           
            #accuracy= accuracy+float((predicted==labels1.data).sum())/float(labels1.size(0))
            #print(predicted.shape,labels.shape)
#            for i ,j in zip(labels,predicted):
#                if i==1:
#                    total_1+=1
#                    if i==1 and j==1:
#                        correct_1+=1
#                    
#                    print('测试集p@1的准确率为%f%%'%(correct_1/total_1*100))  
        if (TP+FP)!=0 and (TP+FN)!=0:
            p = float(TP/ (TP + FP))
            r = float(TP / (TP + FN))
            F1 = 2 * r * p /float( (r + p))
            acc = float((TP + TN) / (TP + TN + FP + FN))
            F1=F1*100
            acc=acc*100
            
            print('准确率acc为%f%%,F1为：%f%%,查准率为p：%f%%，召回率R：%f%%'%(acc,F1,p*100,r*100))
        else:
            print('没分对')
        
        #print(np.all(predicted.view(-1,1)==0))
          
        for i in test_loss:
            test_loss_sum+=i
        test_loss_ave=test_loss_sum/len(test_loss)
        test_loss_epoch.append(test_loss_ave)

           
            
    


# In[11]:



train_dataset= func.DiabetesDataset1('D:/bishedata/train_question.npy','D:/bishedata/train_answers.npy','D:/bishedata/WikiQACorpus/WikiQA-train.tsv')

train_loader= DataLoader(dataset=train_dataset,batch_size=128,shuffle=True)

test_dataset=func.DiabetesDataset1('D:/bishedata/test_questions.npy','D:/bishedata/test_answers.npy','D:/bishedata/WikiQACorpus/WikiQA-test.tsv')

test_loader=DataLoader(dataset=test_dataset,batch_size=128,shuffle=True)


    




#import torch.nn as nn
#
p=0.7
model=nn.Sequential(nn.Linear(1536,768),nn.Dropout(p),nn.Sigmoid(),
                    nn.Linear(768,256),nn.Dropout(p),nn.Sigmoid(),
                    nn.Linear(256,128),nn.Dropout(p),nn.Sigmoid(),
                    nn.Linear(128,64),nn.Dropout(p),nn.Sigmoid(),
                    nn.Linear(64,16),nn.Dropout(p),nn.Sigmoid(),
                    nn.Linear(16,4),nn.Dropout(p),nn.Sigmoid(),
                    nn.Linear(4,2),nn.Dropout(p),nn.Sigmoid())


input_size=48
seq_len=32
hidden_size=16
num_layers=3
num_classes=2


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


criterion=torch.nn.CrossEntropyLoss(reduction='mean') #交叉熵的定义，即损失值
optimizer=torch.optim.Adam(model.parameters(),lr=0.1)#优化器的定义，随机梯度下降法,Adam,SGD


# In[13]:


for epoch in range(150):
    train(epoch)
    test()
    ##########在此加上准确率的执行语句
   



# In[14]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.plot(train_loss_epoch,'g',label="train_loss")
plt.title("loss")
plt.xlabel('epoch')
plt.ylabel('cross_entripy')
plt.legend()
plt.show()


plt.plot(test_loss_epoch,'r',label="test_loss")
plt.title("loss")
plt.xlabel('epoch')
plt.ylabel('cross_entripy')
plt.legend()
plt.show()



plt.plot(train_loss_epoch,'g',label="train_loss")
plt.plot(test_loss_epoch,'r',label="test_loss")
plt.title("loss")
plt.xlabel('epoch')
plt.ylabel('cross_entripy')
plt.legend()
plt.show()