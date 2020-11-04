# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:26:00 2020

@author: 龙
"""
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
