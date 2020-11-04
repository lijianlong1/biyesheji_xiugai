# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:34:34 2020

@author: 龙
"""
#对文件的数据进行整理，先将数据存储，以免以后再训练时的时间开销
import pandas as pd
import numpy as np
from bert_serving.client import BertClient
#from termcolor import colored
import time
time0=time.time()
file_name='D:/bishedata/WikiQACorpus/WikiQA-test.tsv'
out_dir_0='D:/bishedata/test_questions.npy'
out_dir_1='D:/bishedata/test_answers.npy'
test_data=pd.read_csv(file_name, sep='\t', header=0)
print(test_data.head())
print(test_data.shape)
questions=test_data['Question'].tolist()
sentences=test_data['Sentence'].tolist()
print('question and Sentence loaded')
#print(sentences[0:4])
time1=time.time()
print('读取时间为：%f秒'%(time1-time0))
with BertClient(check_length=False) as bc:
    #测试集问题转码并存储
    doc_vecs_0=bc.encode(questions)
    time2=time.time()
    print('转码用的时间为：%f分钟'%((time2-time1)/60))
    print(type(doc_vecs_0))
    print('问题转码完成')
    time3=time.time()
    np.save(out_dir_0,doc_vecs_0)
    print('问题编码存储完成:%f秒'%(time3-time2))
    #测试集的答案存储
    
    
    
    doc_vecs_1=bc.encode(sentences)
    time4=time.time()
    print('转码用的时间为：%f分钟'%((time2-time1)/60))
    print(type(doc_vecs_1))
    print('问题转码完成')
    time5=time.time()
    np.save(out_dir_1,doc_vecs_1)
    print('问题编码存储完成:%f秒'%(time5-time4))

