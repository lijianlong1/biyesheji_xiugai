# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 22:02:45 2020

@author: 龙
"""
#导入相应的包
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.autograd import Variable
import time
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings("ignore")
 
# =============================================================================
# 数据处理数据应该单独送入model中进行训练
# =============================================================================
train_loss=[]
test_loss=[]
train_loss_epoch=[]
test_loss_epoch=[]
batch_size=512
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DiabetesDataset(Dataset):
    def __init__(self,filepath1,filepath2,filepath3):
        
        list1 = np.load(filepath1)
        list2 = np.load(filepath2)
        #list3 = np.concatenate([list1,list2],axis=1)
        data_y = pd.read_csv(filepath3,sep='\t')
        y_label = data_y['Label']
        y_label = np.array(y_label)
       # y_label = y_label.reshape(len(y_label),1)
        
        ######制作one_hot
        y_label = np.array([1 if i==1 else -1 for i 
                             in y_label]).reshape(len(y_label),-1)
#        y_label= np.array([[1 if i==x else 0 for i in range(2)] for x in
#                            y_label]).reshape(-1,1)        
        self.len = list1.shape[0]
        self.que = torch.from_numpy(list1)
        self.ans = torch.from_numpy(list2)
        self.label = torch.from_numpy(y_label.astype(np.float32)).long()
        
    def __getitem__(self,index):
        
        return self.que[index] , self.ans[index] , self.label[index]
    
    def __len__(self):
        return self.len
# =============================================================================
#################在此实现对数据的采样处理，生成相对比较平衡的的数据，用以在
        ########训练的时候优化损失
class DiabetesDataset_bal(Dataset):
    def __init__(self,filepath1,filepath2,filepath3):
        list1=np.load(filepath1)
        list2=np.load(filepath2)
        list3=np.concatenate([list1,list2],axis=1)
        data_y=pd.read_csv(filepath3,sep='\t')
        y_label=data_y['Label']
        y_label=np.array(y_label).reshape(-1,1)
        #####在此处进行数据的下采样，随机去除一些0标签的量
        rus= RandomUnderSampler(sampling_strategy={0:10000},random_state=0)   # 处理采样的方法
        x_train_sm,y_train_sm =rus.fit_sample(list3,y_label)
        ################在此进行上采样，使得数据达到平衡
        ros=RandomOverSampler(sampling_strategy={1:10000},random_state=0)
        x_ros,y_ros=ros.fit_sample(x_train_sm,y_train_sm)
        
        y_ros = np.array([1 if i==1 else -1 for i 
                             in y_ros]).reshape(len(y_ros),-1)
        
        self.len=x_ros.shape[0]
        self.que=torch.from_numpy(x_ros[:,0:768])
        self.ans=torch.from_numpy(x_ros[:,768:1536])
        self.label=torch.from_numpy(y_ros.astype(np.float32)).long()
        
    def __getitem__(self,index):
        return self.que[index],self.ans[index],self.label[index]
    def __len__(self):
        return self.len








# =============================================================================






# =============================================================================
# 模型的搭建即Modele的搭建:
# =============================================================================
class GRU(torch.nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(GRU,self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.gru=torch.nn.GRU(input_size,hidden_size,num_layers,batch_first=True)
        self.max_polling=torch.nn.MaxPool2d(kernel_size=2)
        self.ave_polling=torch.nn.AvgPool2d(kernel_size=2)
        self.relu=torch.nn.ReLU()
        self.fc=torch.nn.Linear(128*8,256)
    def forward(self,x):
        h0=Variable(torch.zeros(self.num_layers,x.size(0),self.hidden_size))
        h0=h0.to(device)
        o,_=self.gru(x,h0)
        
        o=self.ave_polling(o)
        o=o.view(-1,128*8)
        o=self.fc(o)
        #o=self.relu(o)
        return o
    
    
    
class Model(torch.nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(Model,self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm=torch.nn.LSTM(input_size,hidden_size,num_layers,
                                batch_first=True,bidirectional=False)
        #self.fc=torch.nn.Linear(hidden_size*1,num_classes)
        self.sigmoid=torch.nn.Sigmoid()
        self.softmax=torch.nn.Softmax()
        #self.con=torch.nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
        self.max_polling=torch.nn.MaxPool2d(kernel_size=2)
        self.tanh=torch.nn.Tanh()
        self.fc=torch.nn.Linear(128*8,256)
    def forward(self,x):
        h0=torch.rand(self.num_layers*1,x.size(0),self.hidden_size)
        h0=h0.to(device)
        c0=torch.rand(self.num_layers*1,x.size(0),self.hidden_size)
        c0=c0.to(device)
        out,_=self.lstm(x,(h0,c0))
        #out=self.con(out)
        out=self.max_polling(out)
        out=out.view(-1,128*8)
        out=self.fc(out)
        #out=self.sigmoid(out)
        return out

train_dataset = DiabetesDataset_bal('D:/bishedata/train_question.npy','D:/bishedata/train_answers.npy','D:/bishedata/WikiQACorpus/WikiQA-train.tsv')

train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_dataset = DiabetesDataset('D:/bishedata/test_questions.npy','D:/bishedata/test_answers.npy','D:/bishedata/WikiQACorpus/WikiQA-test.tsv')

test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

#print(test_dataset.__getitem__(0))


hidden_size=256
input_size=48
seq_len=16
num_layers=2
model=GRU(input_size,hidden_size,num_layers)
model = model.to(device)
# =============================================================================
# 构建损失函数和优化器
#1.相似度损失函数cosinembeddingloss
#2.hingeloss  loss={max，margin-SQ正+SQ负}
# =============================================================================
criterion = torch.nn.CosineEmbeddingLoss(margin=0.3,size_average=True,reduction='mean')

optimizer = optim.Adam(model.parameters(),lr=0.01)








# =============================================================================
# 进行训练和测试
# =============================================================================
def train(epoch):
    train_loss_sum=0.0
    
    for batch_idx,data in enumerate(train_loader,0):
        
########################提取出batch中的0,1数据        
        
        #data=for 
        
        
        
##################################################################
        inputs_x1,inputs_x2,target=data
        optimizer.zero_grad()
        
        #######################################适合RNN的shape
        inputs_x1=Variable(inputs_x1.view(-1,seq_len,input_size))
        inputs_x2=Variable(inputs_x2.view(-1,seq_len,input_size))
        target=Variable(target.float().view(-1,1))
        ################################################    
        #加上gpu执行语句
        
        inputs_x1,inputs_x2,target=inputs_x1.to(device),inputs_x2.to(device),target.to(device)
        
       # target=target.long().squeeze()
        
        outputs_x1=model(inputs_x1)
        outputs_x2=model(inputs_x2)
        
        loss=criterion(outputs_x1,outputs_x2,target.float())
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
     test_loss_sum=0.0
     with torch.no_grad():
        for data in test_loader:
            inputs_y1,inputs_y2,labels=data
            
            ######将数据转换成适合RNN的shape#####
            inputs_y1=Variable(inputs_y1.view(-1,seq_len,input_size))
            inputs_y2=Variable(inputs_y2.view(-1,seq_len,input_size))
            labels=Variable(labels.view(-1,1))
            
            ######################################
            inputs_y1,inputs_y2,labels=inputs_y1.to(device),inputs_y2.to(device),labels.to(device)
            outputs_y1=model(inputs_y1)
            outputs_y2=model(inputs_y2)
            
            
            #predicted=outputs.data[0]
            #labels1=labels.squeeze()#将其中的一个为一的维度去掉
            loss=criterion(outputs_y1,outputs_y2,labels.float())
            
            test_loss.append(loss.item())
            
            
        for i in test_loss:
            test_loss_sum+=i
        test_loss_ave=test_loss_sum/len(test_loss)
        test_loss_epoch.append(test_loss_ave)


###################在此可以对模型的准确率进行输出和查看模型训练的情况
def evaluate(epoch):
    pass







#if __name__ == '__main__':
#    for epoch in range(100):
#        
#        train(epoch)
#        test()
#        evaluate()

for epoch in range(1):
    time_start=time.time()
    train(epoch)
    test()
    time_now=time.time()
    print('这是第%d次训练过程,训练所需的时间为%f秒'%(epoch+1,((time_now-time_start))))
    print('训练的损失值为%f,测试的损失值为%f'%(train_loss_epoch[epoch],test_loss_epoch[epoch]))


# =============================================================================
# loss图像的绘制
# =============================================================================

import matplotlib.pyplot as plt
plt.plot(train_loss_epoch,'g',label="train_loss")
plt.title("loss")
plt.xlabel('epoch')
plt.ylabel("cosinembeddingloss")
plt.legend()
plt.show()


plt.plot(test_loss_epoch,'r',label="test_loss")
plt.title("loss")
plt.xlabel('epoch')
plt.ylabel("cosinembeddingloss")
plt.legend()
plt.show()



plt.plot(train_loss_epoch,'g',label="train_loss")
plt.plot(test_loss_epoch,'r',label="test_loss")
plt.title("loss")
plt.xlabel('epoch')
plt.ylabel("cosinembeddingloss")
plt.legend()
plt.show()






model_path='D:/biyesheji/model_save/cosin_GRUmodelpara256_10.pth'

torch.save(model, model_path)
#torch.save(model.state_dict(), 'D:/biyesheji/model_save/cosin_BiLSTMmodelpara256_50-2dic.pth')