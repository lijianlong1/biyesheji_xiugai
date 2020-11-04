# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 13:10:01 2020

@author: 龙
"""

# coding: utf-8

# In[44]:


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from __pycache__ import function as func
from torch.autograd import Variable
# 定义几个全局变量

# In[45]:


train_loss=[]
test_loss=[]
train_loss_epoch=[]
test_loss_epoch=[]


# In[46]:



    
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

        
#=========================定义训练函数，以备调用======================
def train(epoch):
    train_loss_sum=0.0
    
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target=data
        optimizer.zero_grad()
        
        #######################################适合RNN的shape
        inputs=Variable(inputs.view(-1,seq_len,input_size))
        target=Variable(target)
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
            images=Variable(images.view(-1,seq_len,input_size))
            labels=Variable(labels)
            
            ######################################
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            
            
            _,predicted=outputs.data.max(dim=1)
            labels1=labels.squeeze()
            loss=criterion(outputs,labels1)
            
            test_loss.append(loss.item())
            #labels=torch.topk(labels,1)[1].squeeze(1)
            
            #prediction = predicted.cpu().numpy()
            #print(np.all(prediction.reshape(-1,1)==0))
            # TP predict 和 label 同时为1
            TP += float(((predicted == 1) & (labels1.data == 1)).cpu().sum())
# TN predict 和 label 同时为0
            TN += float(((predicted == 0) & (labels1.data == 0)).cpu().sum())
            FN += float(((predicted == 0) & (labels1.data == 1)).cpu().sum())
# FP predict 1 label 0)
            FP += float(((predicted == 1) & (labels1.data == 0)).cpu().sum())
            #print(TP,TN,FN,FP)
           
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



train_dataset= func.DiabetesDataset3('D:/bishedata/train_questions.npy','D:/bishedata/train_answers.npy','D:/bishedata/WikiQACorpus/WikiQA-train.tsv')

train_loader= DataLoader(dataset=train_dataset,batch_size=128,shuffle=True)

test_dataset=func.DiabetesDataset('D:/bishedata/test_questions.npy','D:/bishedata/test_answers.npy','D:/bishedata/WikiQACorpus/WikiQA-test.tsv')

test_loader=DataLoader(dataset=test_dataset,batch_size=128,shuffle=True)


    




#import torch.nn as nn
#
#p=0.95
#model=nn.Sequential(nn.Linear(1536,768),nn.Dropout(p),nn.ReLU(),
#                    nn.Linear(768,256),nn.Dropout(p),nn.ReLU(),
#                    nn.Linear(256,128),nn.Dropout(p),nn.ReLU(),
#                    nn.Linear(128,64),nn.Dropout(p),nn.ReLU(),
#                    nn.Linear(64,16),nn.Dropout(p),nn.ReLU(),
#                    nn.Linear(16,4),nn.Dropout(p),nn.ReLU(),
#                    nn.Linear(4,2))


input_size=24
seq_len=64
hidden_size=16
num_layers=3
num_classes=2

model =func.LSTM(input_size,hidden_size,num_layers,num_classes)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

#尝试惩罚权重参数weight=torch.from_numpy(np.array([1.0/95.0,1.0/5.0])).float().to(device),
criterion=torch.nn.CrossEntropyLoss(size_average=True) #交叉熵的定义，即损失值
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)#优化器的定义，随机梯度下降法,Adam,SGD


# In[13]:


for epoch in range(20):
    train(epoch)
    test()
    ##########在此加上准确率的执行语句

#state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}  
#model_path='D:/biyesheji/model_save/BiLSTMmodelpara20_3.pth'
#torch.save(model, model_path)
#torch.save(model.state_dict(), 'D:/biyesheji/model_save/BiLSTMmodelpara20_2dict.pth')
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
## In[ ]:
#
#
#train_dataset.__len__()
#
#train_dataset.__getitem__(0)   
#zero = 0
#total = 0
#for data in train_load:
#    _,y =data
#    total=total+y.size(0)
#    zero = zero+(y==0).sum().item()
#print( '数据样本中的零的出现概率：',(zero/total*100)
#










