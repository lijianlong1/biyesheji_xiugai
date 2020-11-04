# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 21:22:03 2020

@author: 龙
"""
#对文件的数据进行整理，先将数据存储，以免以后再训练时的时间开销
import pandas as pd
import numpy as np
from bert_serving.client import BertClient
#from termcolor import colored
import time
time0=time.time()
file_name='D:/bishedata/WikiQACorpus/WikiQA-train.tsv'
out_dir='D:/bishedata/train_answers.npy'
train=pd.read_csv(file_name, sep='\t', header=0)
print(train.head())
print(train.shape)

sentences=train['Sentence'].tolist()
print('Sentence loaded')
print(sentences[0:4])
time1=time.time()
print('读取时间为：%f'%(time1-time0))
with BertClient(check_length=False) as bc:
    doc_vecs=bc.encode(sentences)
    time2=time.time()
    print('转码用的时间为：%f'%(time2-time1))
    print(type(doc_vecs))
    
    print('问题转码完成')
    time3=time.time()
    np.save(out_dir,doc_vecs)
    print('存储完成:%f'%(time3-time2))
