# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:47:09 2020

@author: 龙
"""
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
#from sklearn import svm
#from sklearn.metrics import hinge_loss
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings("ignore")
#import torch
#torch.cuda.empty_cache()
# =============================================================================
# 数据处理数据应该单独送入model中进行训练
# =============================================================================
train_loss=[]
test_loss=[]
train_loss_epoch=[]
test_loss_epoch=[]
batch_size=32
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
class DiabetesDataset_bal_train(Dataset):
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


class DiabetesDataset_bal_test(Dataset):
    def __init__(self,filepath1,filepath2,filepath3):
        list1=np.load(filepath1)
        list2=np.load(filepath2)
        list3=np.concatenate([list1,list2],axis=1)
        data_y=pd.read_csv(filepath3,sep='\t')
        y_label=data_y['Label']
        y_label=np.array(y_label).reshape(-1,1)
        #####在此处进行数据的下采样，随机去除一些0标签的量
        rus= RandomUnderSampler(sampling_strategy={0:2000},random_state=0)   # 处理采样的方法
        x_train_sm,y_train_sm =rus.fit_sample(list3,y_label)
        ################在此进行上采样，使得数据达到平衡
        ros=RandomOverSampler(sampling_strategy={1:2000},random_state=0)
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
class CosimLoss(torch.nn.Module):
    def __init__(self):
        super(CosimLoss,self).__init__()
        
    def forward(self,ques,ans,label,margin):
        false_cos ,true_cos=0.0,0.0
        count_false,count_true=0,0
        false_cos_ave ,true_cos_ave = 0.0,0.0
        for i,j,k in zip(ques,ans,label):
            
            if torch.equal(k,torch.Tensor([-1]).to(device)): #在此处做判断可能出现问题
               
                false_cos+=torch.cosine_similarity(i.view(1,-1),j.view(1,-1))
                #print(false_cos)
                count_false+=1
               
            if torch.equal(k,torch.Tensor([1]).to(device)):
                true_cos+=torch.cosine_similarity(i.view(1,-1),j.view(1,-1))
                #print(i.view(1,-1).shape,j.shape)
                count_true+=1
        #data=for 
        while count_false==0 or count_true==0:
            print('在这个mini—batch里面全为一个标签')
            continue
        
        false_cos_ave ,true_cos_ave = false_cos/count_false ,true_cos/count_true
        loss = torch.max(torch.Tensor([0]).to(device),margin-true_cos_ave+false_cos_ave)
        #print(true_cos_ave)
        return loss
#loss = max{0,margin-sqt+sqf}
'''
sqt_ave:在一个mini—batch中问题与正确答案的相似度平均
sqf_ave:在一个mini—batch中问题与非正确答案的相似度平均
margin: 训练的偏置，一般在0.2左右建议取值0~0.5之间
'''



# =============================================================================
# 模型的搭建即Modele的搭建:
# =============================================================================
class GRU(torch.nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(GRU,self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.gru=torch.nn.GRU(input_size,hidden_size,num_layers,batch_first=True)
        #self.max_polling=torch.nn.MaxPool2d(kernel_size=128*1)
#        self.avg_polling=torch.nn.AvgPool2d(kernel_size=2)
        self.relu=torch.nn.ReLU()
        self.tanh=torch.nn.Tanh()
        self.sigmoid=torch.nn.Sigmoid()
        #self.fc=torch.nn.Linear(32*6,64)
    def forward(self,x):
        h0=Variable(torch.randn(self.num_layers,x.size(0),self.hidden_size))
        h0=h0.to(device)
        o,_=self.gru(x,h0)
        
        #o=self.max_polling(o)
        
        #o=o.view(-1,16)
        #o=self.fc(o)
        o=o[:,-1,:]
        o=self.relu(o)#############在此可以尝试其他的激活函数
        return o
    
    
    
class Model(torch.nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(Model,self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm=torch.nn.LSTM(input_size,hidden_size,num_layers,
                                batch_first=True,bidirectional=True)
        #torch.nn.Transformer
        #尝试使用transformer
        #self.fc=torch.nn.Linear(hidden_size*1,num_classes)
        self.sigmoid=torch.nn.Sigmoid()
        self.softmax=torch.nn.Softmax()
        #self.con=torch.nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
        #self.max_polling=torch.nn.MaxPool2d(kernel_size=2)
        self.tanh=torch.nn.Tanh()
        self.relu=torch.nn.ReLU()
        #self.fc=torch.nn.Linear(100*12,256)
    def forward(self,x):
        h0=torch.randn(self.num_layers*2,x.size(0),self.hidden_size)
        h0=h0.to(device)
        c0=torch.randn(self.num_layers*2,x.size(0),self.hidden_size)
        c0=c0.to(device)
        out,_=self.lstm(x,(h0,c0))
        #out=self.con(out)
        #out=self.max_polling(out)
        out=out[:,-1,:]
        #out=out.view(-1,100*12)
        #out=self.fc(out)
        
        out=self.relu(out)
        return out

train_dataset = DiabetesDataset_bal_train('D:/bishedata/train_questions.npy','D:/bishedata/train_answers.npy','D:/bishedata/WikiQACorpus/WikiQA-train.tsv')

train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_dataset = DiabetesDataset_bal_test('D:/bishedata/test_questions.npy','D:/bishedata/test_answers.npy','D:/bishedata/WikiQACorpus/WikiQA-test.tsv')

test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

#print(test_dataset.__getitem__(0))


hidden_size=512
input_size=768
seq_len=1
num_layers=1
model=Model(input_size,hidden_size,num_layers)
model = model.to(device)
# =============================================================================
# 构建损失函数和优化器
#1.相似度损失函数cosinembeddingloss
#2.hingeloss  loss={max，margin-SQ正+SQ负}
# =============================================================================
#criterion = torch.nn.CosineEmbeddingLoss(margin=0.55,size_average=True,reduction='mean')

criterion = CosimLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)








# =============================================================================
# 进行训练和测试
# =============================================================================
def train(epoch):
    train_loss_sum=0.0
    
    for batch_idx,data in enumerate(train_loader,0):
        #在batch里面定义一些变量空间
        inputs_x1,inputs_x2,target=data
##################################################################
        
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
        
#####################################################################        
        loss=criterion(outputs_x1,outputs_x2,target,margin=0.2)
#        loss_tensor=torch.Tensor([loss])
#        loss=Variable(loss_tensor,requires_grad=True)
        
        loss.backward()
        optimizer.step()
        #running_loss+=loss.item()
        #print(loss.item())
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
            loss=criterion(outputs_y1,outputs_y2,labels.float(),margin=0.2)
            
            test_loss.append(loss.item())
            
            
        for i in test_loss:
            test_loss_sum+=i
        test_loss_ave=test_loss_sum/len(test_loss)
        test_loss_epoch.append(test_loss_ave)


###################在此可以对模型的准确率进行输出和查看模型训练的情况
def evaluate():
    load_que='D:/bishedata/test_questions.npy'  #读取问题数组
    load_ans='D:/bishedata/test_answers.npy'
    list1=np.load(load_que)
    list2=np.load(load_ans)
    #print(type(list1))
#print(list1.shape,list2.shape)
    data_y = pd.read_csv('D:/bishedata/WikiQACorpus/WikiQA-test.tsv',sep='\t')
    y_label = data_y['Label']
    ques_id = data_y['QuestionID']

    
#################################
#    seq_len = 24
#    input_size = 32
    with torch.no_grad():    
############将答案和问题数组一对一对的送入到model中取得相似度向量并存储好####
#    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        score=[]
#    score_768=[]
        for i,j in zip(list1,list2):
            i_view=torch.from_numpy(i).view(-1,seq_len,input_size)
            j_view=torch.from_numpy(j).view(-1,seq_len,input_size)
            out_0 = model(i_view.to(device))
            out_1 = model(j_view.to(device))
            cos = torch.cosine_similarity(out_0.view(1,-1),out_1.view(1,-1))
#        cos_768=torch.cosine_similarity(i_view.to(device).view(1,-1),j_view.to(device).view(1,-1))
            score.append(cos)
#        score_768.append(cos_768)
#    print(type(score),len(score))
        que_num=1
        start=0
        end=0
        true_count=0
        all_count=0
        mrr_list=[]
        map_batch_sum=0.0
        for index,que in enumerate(ques_id):
            if index+1==len(ques_id):
                break
            if ques_id[index]!=ques_id[index+1]:
                que_num+=1
                end=index+1
#            print(ques_id[start:end])
        #############在下面进行batch问题的计算与处理#
                list_batch=[]
                for i,j,k in zip(ques_id[start:end],score[start:end],y_label[start:end]):
                    a=(i,j.item(),k)
                    list_batch.append(a)
                    list_sortby_score=sorted(list_batch, key=lambda s: s[1], reverse=True)
                if list_sortby_score[0][2]==1:
                    true_count+=1
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
                for index,i in enumerate(c):
                    if int(i)==1:
                        map_count+=1
                        map_1+=map_count/(index+1)
                        map_batch=map_1/map_count
                if((np.array(y_label[start:end]) == 1).any()):
                        all_count+=1
                        map_batch_sum+=map_batch
#       
                start=end
###########最后面的一个问题单独提取出来，因为上面没有统计到这个问题
   
#print(ques_id[end:len(ques_id)])
#最后一个问题单独的提取出来，已经记录下指针end的位置      
        if((np.array(y_label[end:len(ques_id)])==1).any()):
            all_count+=1
#    print(true_count,all_count,que_num)
#    print(true_count/all_count)
    return true_count/all_count,sum(mrr_list)/all_count,map_batch_sum/all_count







#if __name__ == '__main__':
#    for epoch in range(100):
#        
#        train(epoch)
#        test()
#        evaluate()

for epoch in range(30):
    time_start=time.time()
    train(epoch)
    test()
    #############acc,map,mrr计算########
    acc,mrr,map_total=evaluate()
    ###################################
    time_now=time.time()
    print('这是第%d次训练过程,训练所需的时间为%f秒'%(epoch+1,((time_now-time_start))))
    print('训练的损失值为%f,测试的损失值为%f'%(train_loss_epoch[epoch],test_loss_epoch[epoch]))
    print('模型的准确率为acc=%f,mrr的值为：%f,map的值为：%f'%(acc,mrr,map_total))
    #print('训练的损失值为%f'%(train_loss_epoch[epoch]))
    ###############直接写一个算法
    
# =============================================================================
# loss图像的绘制
# =============================================================================

import matplotlib.pyplot as plt
#plt.plot(train_loss_epoch,'g',label="train_loss")
#plt.title("loss")
#plt.xlabel('epoch')
#plt.ylabel("hingeloss")
#plt.legend()
#plt.show()
#
#
#plt.plot(test_loss_epoch,'r',label="test_loss")
#plt.title("loss")
#plt.xlabel('epoch')
#plt.ylabel("hingeloss")
#plt.legend()
#plt.show()
plt.plot(train_loss_epoch,'g',label="train_loss")
plt.plot(test_loss_epoch,'r',label="test_loss")
plt.title("loss")
plt.xlabel('epoch')
plt.ylabel("hingeloss")
plt.legend()
plt.show()






#model_path='D:/biyesheji/model_save/cosin_gru_modelpara_hinge_30_70map.pth'
#
#torch.save(model, model_path)
#torch.save(model.state_dict(), 'D:/biyesheji/model_save/cosin_BiLSTMmodelpara256_50-2dic.pth')
print('input_size:',input_size,'batch-size=',batch_size,'sequence_len=',seq_len,'使用的是gru网络训练简化训练30轮')