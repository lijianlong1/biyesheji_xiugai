# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:12:07 2020

@author: 龙
"""

import numpy as np
import pandas as pd
import torch
# =============================================================================
# 尝试将测试集的问题和答案进行打分，并存储在
# =============================================================================
#在此时读取答案和问题的numpy数组
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
        #self.max_polling=torch.nn.MaxPool2d(kernel_size=2)
        self.tanh=torch.nn.Tanh()
        self.relu=torch.nn.ReLU()
        #self.fc=torch.nn.Linear(100*12,256)
    def forward(self,x):
        h0=torch.rand(self.num_layers*1,x.size(0),self.hidden_size)
        h0=h0.to(device)
        c0=torch.rand(self.num_layers*1,x.size(0),self.hidden_size)
        c0=c0.to(device)
        out,_=self.lstm(x,(h0,c0))
        #out=self.con(out)
        #out=self.max_polling(out)
        out=out[:,-1,:]
        #out=out.view(-1,100*12)
        #out=self.fc(out)
        
        out=self.tanh(out)
        return out


load_que='D:/bishedata/train_questions.npy'  #读取问题数组
load_ans='D:/bishedata/train_answers.npy'
list1=np.load(load_que)
list2=np.load(load_ans)
print(type(list1))
print(list1.shape,list2.shape)
data_y = pd.read_csv('D:/bishedata/WikiQACorpus/WikiQA-train.tsv',sep='\t')
y_label = data_y['Label']
ques_id = data_y['QuestionID']

print(type(list1),type(ques_id))
print(list1.shape,list2.shape,ques_id.shape)
#################################
seq_len = 12
input_size = 64
torch.no_grad()
dir_file="D:/biyesheji/model_save/cosin_lstm_modelpara_hinge_30.pth"
model = torch.load(dir_file) 
model = model.eval()     
############将答案和问题数组一对一对的送入到model中取得相似度向量并存储好####
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
score=[]
score_768=[]
for i,j in zip(list1,list2):
    i_view=torch.from_numpy(i).view(-1,seq_len,input_size)
    j_view=torch.from_numpy(j).view(-1,seq_len,input_size)
    out_0 = model(i_view.to(device))
    out_1 = model(j_view.to(device))
    cos = torch.cosine_similarity(out_0.view(1,-1),out_1.view(1,-1))
    cos_768=torch.cosine_similarity(i_view.to(device).view(1,-1),j_view.to(device).view(1,-1))
    score.append(cos)
    score_768.append(cos_768)
print(type(score),len(score))
#n=2
#for i,j,k in zip(ques_id,score[0:300],y_label):
#    print(n ,i, j.item(),k)
#    n=n+1
#list10=[]
#a=(1,2,3)
#b=(1,2,4)
#list10.append(a)
#list10.append(b)
#print(list10[0][2])
que_num=1
start=0
end=0
true_count=0
all_count=0
mrr_list=[]
map_list=[]
map_batch_sum=0.0
for index,que in enumerate(ques_id):
    if index+1==len(ques_id):
        break
    if ques_id[index]!=ques_id[index+1]:
        que_num+=1
        end=index+1
        #print(ques_id[start:end])
        #############在下面进行batch问题的计算与处理#
        list_batch=[]
        for i,j,k in zip(ques_id[start:end],score[start:end],y_label[start:end]):
            a=(i,j.item(),k)
            list_batch.append(a)
        list_sortby_score=sorted(list_batch, key=lambda s: s[1], reverse=True)
        if list_sortby_score[0][2]==1:
            true_count+=1
#        b=np.array(list_sortby_score[:][:])
#        print(b[:,2])
        #break
        #转化成数组进行操作
        b=np.array(list_sortby_score[:][:])
        c=b[:,2].tolist()
      
   # print(c,type(int(c[1])))
        #print(b,c,list_sortby_score[:][:],len(mrr_list))
        for index,i in enumerate(c):
            if int(i)==1:
                mrr=1.0/(index+1)
                mrr_list.append(mrr)
                break
        map_count=0
        map_1=0.0
        map_ave=[]
        
        for index,i in enumerate(c):
            if int(i)==1:
                map_count+=1
                map_1+=map_count/(index+1)
                map_batch=map_1/map_count
       
    #print(map_ave)
        
                
                  
               
            
        if((np.array(y_label[start:end]) == 1).any()):
            all_count+=1
            map_batch_sum+=map_batch
#        b=y_label[start:end]
#        b=np.array(b)
#        print(type(b),b)
#        break
          
        
        ###############
        start=end
###########最后面的一个问题单独提取出来，因为上面没有统计到这个问题
   
#print(ques_id[end:len(ques_id)])
#最后一个问题单独的提取出来，已经记录下指针end的位置      
if((np.array(y_label[end:len(ques_id)])==1).any()):
    all_count+=1
print(true_count,all_count,que_num)
print(true_count/all_count,sum(mrr_list)/all_count,map_batch_sum/all_count)

#    else:
#        start=end+1
    