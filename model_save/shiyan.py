# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:03:55 2020

@author: 龙
"""
import torch
torch.cuda.empty_cache()
import numpy as np
#from sklearn.metrics import hinge_loss
#from torch.autograd import Variable
import math
import warnings
warnings.filterwarnings("ignore")
from torch.autograd import Variable
 
#class Model(torch.nn.Module):
#    def __init__(self,input_size,hidden_size,num_layers):
#        super(Model,self).__init__()
#        self.hidden_size=hidden_size
#        self.num_layers=num_layers
#        self.lstm=torch.nn.LSTM(input_size,hidden_size,num_layers,
#                                batch_first=True,bidirectional=False)
#        #self.fc=torch.nn.Linear(hidden_size*1,num_classes)
#        self.sigmoid=torch.nn.Sigmoid()
#        self.softmax=torch.nn.Softmax()
#        #self.con=torch.nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
#        self.max_polling=torch.nn.MaxPool2d(kernel_size=2)
#        
#        self.fc=torch.nn.Linear(16*16,200)
#        self.relu=torch.nn.ReLU()
#    def forward(self,x):
#        h0=torch.randn(self.num_layers*1,x.size(0),self.hidden_size)
#        h0=h0.to(device)
#        c0=torch.randn(self.num_layers*1,x.size(0),self.hidden_size)
#        c0=c0.to(device)
#        out,_=self.lstm(x,(h0,c0))
#        #out=self.con(out)
#        out=self.max_polling(out)
#        out=out.view(batch_size,-1)
#        out=self.fc(out)
#        out=self.relu(out)
#        return out
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
        
        o=self.max_polling(o)
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
        #out=self.max_polling(out)
        out=out[:,-1,:]
        
        #out=self.sigmoid(out)
        return out

def softmax(x):
    exp_x = np.exp(x)
    result = exp_x/np.sum(exp_x)
    return result 

def cos_sim(x,y):
    #输入的x，y应为一个np数组
    cos=np.dot(x,y)/(math.sqrt(np.dot(x,x))*math.sqrt(np.dot(y,y)))
    return cos





load_que='D:/bishedata/test_questions.npy'  #读取问题数组
load_ans='D:/bishedata/test_answers.npy'
list1=np.load(load_que)
list2=np.load(load_ans)
#list3=np.concatenate([list1,list2],axis=1)
x_data=list1
x0=x_data[5]

y_data=list2
y0=y_data[5]





# =============================================================================
# 在下面直接对x0，和y0数据进行读取和修改，同时进行打分判定
# =============================================================================













x0=torch.from_numpy(x0)
y0=torch.from_numpy(y0)


data1 = x0.tolist()
data2 = y0.tolist()



c=cos_sim(data1,data2)
print('在还没输入模型时的数据类型为',type(data1))
print(c)

#hidden_size=128
input_size=64
seq_len=12
#num_layers=3
batch_size=1
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
###########################################################################
model_dir="D:/biyesheji/model_save/cosin_GRUmodelpara_hinge_50.pth"
torch.no_grad()
model1=torch.load(model_dir)

model1=model1.eval()


# =============================================================================
#使用dict加载模型
# the_model = TheModelClass(*args, **kwargs)
# the_model.load_state_dict(torch.load(PATH))
# =============================================================================
############################################################################
# =============================================================================
#model1 =GRU(input_size,hidden_size,num_layers=2)
# =============================================================================
model1=model1.to(device)
torch.no_grad()
x1=x0.view(-1,seq_len,input_size)
y1=y0.view(-1,seq_len,input_size)
x1=x1.to(device)
y1=y1.to(device)
out=model1(x1)
torch.no_grad()
out0=model1(y1)
torch.no_grad()
#soft_out=out.data.cpu().numpy()
#soft_out=softmax(soft_out)
print(out.shape,torch.cosine_similarity(out,out0))
#pre,predicted=out.data.max(dim=1)
print('输入后的数据类型为',type(out))

que=out.tolist()
ans=out0.tolist()
#print(len(que),que)
#print(que)
#ques=np.array(que)
#answ=np.array(ans)
for i,j in zip(que,ans):
    ques=i
    answ=j
    #print(ques,j)
cos=cos_sim(ques,answ)
print(cos)
#print(cos_sim(out.tolist(),out0.tolist()))
#print(type(out),type(que),type(ques))
#x=12.31
#y=torch.Tensor([x])
#print(y/2)
#print(3.0/12)
#print(i,j)
#print(soft_out)
#print(predicted.item())

