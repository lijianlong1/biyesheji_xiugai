# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 09:06:41 2020

@author: 龙
"""
# =============================================================================
# 数据读取模块
import numpy as np
import pandas as pd
import time
import torch 
from imblearn.over_sampling import SMOTE 
# 定义几个全局变量
from imblearn.under_sampling import RandomUnderSampler
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
load_que='D:/bishedata/train_question.npy'  #读取问题数组
load_ans='D:/bishedata/train_answers.npy'
list1=np.load(load_que)
list2=np.load(load_ans)
list3=np.concatenate([list1,list2],axis=1)
print(list3.shape)

data=pd.read_csv('D:/bishedata/WikiQACorpus/WikiQA-train.tsv', sep='\t')
print(data.head(5))
y_lable=data['Label']

y_lable=np.array(y_lable)
y_lable=y_lable.reshape(len(y_lable),1)


sm = SMOTE(random_state = 0)    # 处理过采样的方法
x_data,y_lable = sm.fit_sample(list3,y_lable)


x_data=torch.from_numpy(x_data).to(device)

y_data=torch.from_numpy(y_lable.astype(np.float32)).to(device)



load_test_que='D:/bishedata/test_questions.npy'  #读取问题数组
load_test_ans='D:/bishedata/test_answers.npy'
test_list1=np.load(load_test_que)
test_list2=np.load(load_test_ans)
test_list3=np.concatenate([test_list1,test_list2],axis=1)


data_test=pd.read_csv('D:/bishedata/WikiQACorpus/WikiQA-test.tsv', sep='\t')

y_test_lable=data_test['Label']

y_test_lable=np.array(y_test_lable)
y_test_lable=y_test_lable.reshape(len(y_test_lable),1)
x_test_data=torch.from_numpy(test_list3).to(device)

y_test_data=torch.from_numpy(y_test_lable.astype(np.float32)).to(device)


# =============================================================================
#线性模块的定义：
# =============================================================================

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(1536,768)
        self.linear2 = torch.nn.Linear(768,256)
        self.linear3 = torch.nn.Linear(256,64)
        self.linear4 = torch.nn.Linear(64,16)
        self.linear5 = torch.nn.Linear(16,4)
        self.linear6 = torch.nn.Linear(4,1)
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
model=Model()
model=model.to(device)
criterion=torch.nn.BCELoss(size_average=True) #交叉熵的定义，即损失值
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)#优化器的定义，随机梯度下降法

learn_history=[]
test_history=[]
###########开始训练数据
for epoch in range(100):
    #前项传播算法
    since=time.time()
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss.item())
    #后向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()#进行梯度更新
    end=time.time()
    print('训练所用的时间为%fs'%(end-since))
    learn_history.append(loss.item())
    
    print('###################测试的相应的数据如下：##########')
    with torch.no_grad():
        start1=time.time()
        y_test_pre=model(x_test_data)
        loss_test=criterion(y_test_pre,y_test_data)
        print(epoch,loss_test.item())
        end1=time.time()
        print('训练所用的时间为%fs'%(end1-start1))
        test_history.append(loss_test.item())
#图形的绘制
model_path='D:/biyesheji/model_save/Linearmodelpara17.pth'
torch.save(model, model_path)
#%matplotlib inline
import matplotlib.pyplot as plt
l=learn_history
plt.plot(l,'g',label="train_loss")
plt.title("loss")
plt.xlabel('epoch')
plt.ylabel('cross_entripy')
plt.legend()
plt.show()

t=test_history
plt.plot(t,'r',label="test_loss")
plt.title("loss")
plt.xlabel('epoch')
plt.ylabel('cross_entripy')
plt.legend()
plt.show()

l=learn_history
plt.plot(l,'g',label="train_loss")
plt.plot(t,'r',label="test_loss")
plt.title("loss")
plt.xlabel('epoch')
plt.ylabel('cross_entripy')
plt.legend()
plt.show()











