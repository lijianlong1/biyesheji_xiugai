# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 23:54:03 2020

@author: 龙
"""
import numpy as np
import pandas as pd
import torch


def eva():
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
    seq_len = 24
    input_size = 32
    torch.no_grad()    
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
    que_num=1
    start=0
    end=0
    true_count=0
    all_count=0
    for index,que in enumerate(ques_id):
        if index+1==len(ques_id):
            break
        if ques_id[index]!=ques_id[index+1]:
            que_num+=1
            end=index+1
            print(ques_id[start:end])
        #############在下面进行batch问题的计算与处理#
            list_batch=[]
            for i,j,k in zip(ques_id[start:end],score[start:end],y_label[start:end]):
                a=(i,j.item(),k)
                list_batch.append(a)
                list_sortby_score=sorted(list_batch, key=lambda s: s[1], reverse=True)
            if list_sortby_score[0][2]==1:
                true_count+=1
            if((np.array(y_label[start:end]) == 1).any()):
                all_count+=1
#       
            start=end
###########最后面的一个问题单独提取出来，因为上面没有统计到这个问题
   
#print(ques_id[end:len(ques_id)])
#最后一个问题单独的提取出来，已经记录下指针end的位置      
    if((np.array(y_label[end:len(ques_id)])==1).any()):
        all_count+=1
#    print(true_count,all_count,que_num)
#    print(true_count/all_count)
    return true_count/all_count

