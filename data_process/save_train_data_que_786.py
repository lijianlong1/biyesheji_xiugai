# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 21:15:55 2020

@author: 龙
"""
import pandas as pd
import numpy as np
from bert_serving.client import BertClient
from termcolor import colored
file_name='D:/bishedata/WikiQACorpus/WikiQA-train.tsv'
out_dir='D:/bishedata/train_question.npy'
train=pd.read_csv(file_name, sep='\t', header=0)
print(train.head())
print(train.shape)

questions=train['Question']#.tolist()
print('Sentence loaded')
print(questions[0:4])
with BertClient(check_length=False) as bc:
    doc_vecs=bc.encode(questions)
    print(type(doc_vecs))
    print('问题转码完成')
    np.save(out_dir,doc_vecs)
    print('存储完成')