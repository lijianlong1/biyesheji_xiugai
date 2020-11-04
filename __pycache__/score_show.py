import tkinter as tk  # 使用Tkinter前需要先导入

from torch.autograd import Variable
from __pycache__ import translate as tr
from bert_serving.client import BertClient
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
        
        #out=self.tanh(out)
        return out
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
# 第1步，实例化object，建立窗口window
window = tk.Tk()
 
# 第2步，给窗口的可视化起名字
window.title('答案选取模型的展示:@author: 龙')
 
# 第3步，设定窗口的大小(长 * 宽)
window.geometry('800x400')  # 这里的乘是小x
 
l = tk.Label(window, text='欢迎！这是答案选取模型的展示', font=('Calibri', 15), width=30, height=2)
l.pack()

copy_right = tk.Label(window, text='copyright:lijianlong', font=('Arial',10), width=30, height=1)
copy_right.place(x=600,y=380)
# 使用label提示用户输入相关的信息
tk.Label(window, text='请输入问题', font=('Arial', 10)).place(x=50, y=60)
tk.Label(window, text='请输入答案1', font=('Arial',10)).place(x=50, y=100)
tk.Label(window, text='请输入答案2', font=('Arial',10)).place(x=50,y=140)
tk.Label(window, text='请输入答案3', font=('Arial',10)).place(x=50,y=180)
tk.Label(window, text='请输入答案4', font=('Arial',10)).place(x=50,y=220)
# 第6步，答案问题输入框entry
# 问题输入框
ques = tk.StringVar()
ques.set('北京的面积有多大？')
ques = tk.Entry(window, textvariable=ques, font=('Arial', 10),width='25')
ques.place(x=140,y=60)
# 答案输入框
ans_1 = tk.StringVar()
ans_1.set('北京是中国的首都。')
ans_1 = tk.Entry(window, textvariable=ans_1, font=('Arial', 10),width='25')
ans_1.place(x=140,y=100)



ans_2 = tk.StringVar()
ans_2.set('北京在中国的北边。')
ans_2 = tk.Entry(window, textvariable= ans_2, font=('Arial', 10),width='25')
ans_2.place(x=140,y=140)



ans_3 = tk.StringVar()
ans_3.set('北京的面积为2万平方公里。')
ans_3 = tk.Entry(window, textvariable=ans_3, font=('Arial', 10),width='25')
ans_3.place(x=140,y=180)



ans_4 = tk.StringVar()
ans_4.set('今天的天气很好。')
ans_4 = tk.Entry(window, textvariable=ans_4, font=('Arial', 10),width='25')
ans_4.place(x=140,y=220)



##################################定义答案问题打分功能与显示功能###############
def score_show():
    get_ques = ques.get()
    get_ans_1 =ans_1.get()
    get_ans_2 =ans_2.get()
    get_ans_3 =ans_3.get()
    get_ans_4 =ans_4.get()
    with BertClient(check_length=False) as bc:
        ####################实现系统的人性化设置，自动的将中文翻译成英语问题#######
        q = tr.translator(get_ques)
        a_1 = tr.translator(get_ans_1)
        a_2 = tr.translator(get_ans_2)
        a_3 = tr.translator(get_ans_3)
        a_4 = tr.translator(get_ans_4)
        ################在这个地方拿到答案和问题的bert编码##########
        q1 = torch.from_numpy(bc.encode([q]))
        a1 = torch.from_numpy(bc.encode([a_1]))
        a2 = torch.from_numpy(bc.encode([a_2]))
        a3 = torch.from_numpy(bc.encode([a_3]))
        a4 = torch.from_numpy(bc.encode([a_4]))
        #######################################################
        #在此加载训练好的模型进行实际的测试拿到问题和答案的相似度打分#
        seq_len = 1
        input_size = 768
        torch.no_grad()
        dir_file="D:/biyesheji/model_save/cosin_gru_modelpara_hinge_30_70map.pth"
        model = torch.load(dir_file)
        model = model.eval()
        #cos_0 = torch.cosine_similarity(q1.view(1,-1),a1.view(1,-1))
        q1 = q1.view(-1,seq_len,input_size)
        a1 = a1.view(-1,seq_len,input_size)
        a2 = a2.view(-1,seq_len,input_size)
        a3 = a3.view(-1,seq_len,input_size)
        a4 = a4.view(-1,seq_len,input_size) 
# =============================================================================
#在此将句子向量输入到模型中，得到训练好的向量
        
        out_0 = model(q1.to(device))
        out_1 = model(a1.to(device))
        out_2 = model(a2.to(device))
        out_3 = model(a3.to(device))
        out_4 = model(a4.to(device))
        cos_1 = torch.cosine_similarity(out_0.view(1,-1),out_1.view(1,-1))
        cos_2 = torch.cosine_similarity(out_0.view(1,-1),out_2.view(1,-1))
        cos_3 = torch.cosine_similarity(out_0.view(1,-1),out_3.view(1,-1))
        cos_4 = torch.cosine_similarity(out_0.view(1,-1),out_4.view(1,-1))
        
        
        
        cos_list=[('答案1',cos_1.item()),('答案2',cos_2.item()),('答案3',cos_3.item()),('答案4',cos_4.item())]
        sort_cos_list=sorted(cos_list, key=lambda x:x[1],reverse=True)
        max_score=str(sort_cos_list[0][1])
        prefer_ans='推荐您选择'+sort_cos_list[0][0]+',其与问题的相似度为：'+max_score
# =============================================================================
        ###########################################################
        
#        l = tk.Label(window, text=cos_0.data, bg='red',fg='black', font=('Arial', 10), width=30, height=2)
#        l.place(x=400,y=60)  
        l = tk.Label(window, text=cos_1.item(), bg='white',fg='black', font=('Arial', 10), width=30, height=1)
        l.place(x=400,y=100) 
        l = tk.Label(window, text=cos_2.item(), bg='white',fg='black', font=('Arial', 10), width=30, height=1)
        l.place(x=400,y=140)   
        l = tk.Label(window, text=cos_3.item(), bg='white',fg='black', font=('Arial', 10), width=30, height=1)
        l.place(x=400,y=180)  
        l = tk.Label(window, text=cos_4.item(), bg='white',fg='black', font=('Arial', 10), width=30, height=1)
        l.place(x=400,y=220)
######################################记录最高的得分，并进行sort排序###########
        rank_ans='第1名：'+str(sort_cos_list[0][0])+'\t'+'第2名：'+str(sort_cos_list[1][0])+'\t'+'第3名：'+str(sort_cos_list[2][0])+'\t'+'第4名：'+str(sort_cos_list[3][0])
        l = tk.Label(window, text=rank_ans, bg='white',fg='black', font=('Arial', 10), width=100,height=2)
        l.place(x=0,y=280)
        
        
        l = tk.Label(window, text=prefer_ans, bg='white',fg='black', font=('Arial', 10),width=100,height=2)
        l.place(x=0,y=320)
 
 
# 第7步，打分 按钮
btn_1 = tk.Button(window, text='show_scores',bg='white',fg='black', command=score_show,height=1)
btn_1.place(x=400, y=60)





# 第10步，主窗口循环显示
window.mainloop()