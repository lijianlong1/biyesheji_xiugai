# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:11:35 2020

@author: 龙
"""

import numpy as np
import torch
from bert_serving.client import BertClient
import translate as tr
import math
def softmax(x):
    exp_x = np.exp(x)
    result = exp_x/np.sum(exp_x)
    return result 
def cos_sim(x,y):
    #输入的x，y应为一个np数组
    cos=np.dot(x,y)/(math.sqrt(np.dot(x,x))*math.sqrt(np.dot(y,y)))
    return cos



with BertClient(check_length=False) as bc:
    q=tr.translator('北京有多大')
    que=bc.encode([q])
    a=tr.translator('北京有200平方公里')
    print(a)
    ans=bc.encode([a])
    x0=np.concatenate([que,ans],axis=1)
    x0=torch.from_numpy(x0)
    input_size=24
    seq_len=64
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    model_dir='D:/biyesheji/model_save/BiLSTMmodelpara20_3.pth'
    model = torch.load(model_dir)
    
#    the_model = TheModelClass(*args, **kwargs)
#    the_model.load_state_dict(torch.load(PATH))
    model=model.to(device)
    torch.no_grad()
    x0=x0.view(-1,seq_len,input_size)
    x0=x0.to(device)
    out=model(x0)

    #soft_out=out.data.cpu().numpy()
    #soft_out=softmax(soft_out)

    #pre,predicted=out.data.max(dim=1)
    pre=out.data[0][1]
    print(pre)
    score=pre
    print(out)
    print('为正确答案的概率为：%f'%score)
    print(ans.shape)
    que=que.tolist()
    ans=ans.tolist()
    for i,j in zip(que,ans):
        ques=i
        answ=j
        
     
        
    #print(que.reshape(1,768))
    cos=cos_sim(ques,answ)
    print(cos)
    #print(predicted.item())