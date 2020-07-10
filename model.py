import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

def clones(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
#归一化层
class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self,x):
        mean = x.mean(-1,keepdim = True)
        std = x.std(-1,keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) +self.b_2 #a_2和b_2是我们定义的参数，可以在训练中学习。
'''残差网络与dropout'''
class SublayerConnection(nn.Module):
    def __init__(self,size,dropout):
        super(SublayerConnection,self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x))) #先normaliztion在经过multi-head/feed forward层在dropout再残差连接

class Decoder(nn.Module):
    def __init__(self,layer,N):
        super(Decoder,self).__init__()
        self.layers = clones(layer,N)
        #print("这里打印出看下layer.size是",layer.size) #128
        self.norm = LayerNorm(layer.size)
    def forward(self, x,tgt_mask):
        for layer in self.layers:
            x = layer(x,tgt_mask) #每一个layer都通过两次norm规范化，通过两层后，最后在layer normalization一下。
        return self.norm(x)

class DecoderLayer(nn.Module): #包含前馈层等各种的一个解码器层
    '''decoder consist of with three layers, multi-head, feed_forward,src_attn我们这里可能只需要两层'''
    def __init__(self,size,self_attn,feed_forward,dropout):
        super(DecoderLayer,self).__init__()
        self.size = size #这个size是维度512维/128维
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size,dropout),2)
    def forward(self,x,tgt_mask):
        x = self.sublayer[0](x,lambda x:self.self_attn(x,x,x,tgt_mask))
        return self.sublayer[1](x,self.feed_forward)  #在两个子层后进行残差连接和归一化

def attention(query,key,value,mask=None,dropout=None):
    '''compute the attention'''
    d_k = query.size(-1) #最后一个维度的大小，特征维度
    #print("为了使得梯度更稳定:",math.sqrt(d_k*2))
    scores = torch.matmul(query,key.transpose(-2,-1))/ math.sqrt(d_k*2)
    if mask is not None:
        scores = scores.masked_fill(mask==0,-1e9) #将mask中为0的数字所对应的scores位置置换成-1e9
    p_attn = F.softmax(scores,dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn,value),p_attn
class MultiHeadAttention(nn.Module):
    def __init__(self,h,d_model,dropout=0.1):
        super(MultiHeadAttention,self).__init__()
        assert  d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model,d_model),4)
        self.attn = None
        self.dropout = nn.Dropout(p = dropout)
    def forward(self,query,key,value,mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)  #所有head的mask都是相同的，输入是mask(batch,1,seq)所以加一维变成mask(batch,1,1,seq)
        nbatches = query.size(0)
        query,key,value = [l(x).view(nbatches,-1,self.h,self.d_k).transpose(1,2) for l,x in zip(self.linears,(query,key,value))]
        #用三个线性层进行处理，query最初是(batch,seq,128)用view将其变成(batch,seq,4,32)然后再transport。key和value的运算完全相同
        x,self.attn = attention(query,key,value,mask=mask,dropout=self.dropout)
        x = x.transpose(1,2).contiguous().view(nbatches,-1,self.h * self.d_k) #变回(batch,seq,4*32)
        return self.linears[-1](x)
class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.w_1 = nn.Linear(d_model,d_ff)
        self.w_2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len= 5000 ):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(dropout)
        '''compute the positional encodings once in log space'''
        pe = torch.zeros(max_len,d_model)  #5000*128
        position = torch.arange(0.,max_len).unsqueeze(1) #5000*1
        div_term = torch.exp( torch.arange(0.,d_model,2)*
                              -(math.log(10000.0)/d_model )) #64
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #1*5000*128
        self.register_buffer('pe',pe) #缓存
    def forward(self,x):
        # z = self.pe                 #z.shape = [1,5000,128]
        x = x + self.pe[:,:x.size(1)] #x.size(1) 就是seq  这里得到的是[1,seq,128] 而x为[batch_size,seq,128],对每一个batch都加
        return self.dropout(x)

class InputLinears(nn.Module):
    def __init__(self,real_dimension,d_model):
        super(InputLinears,self).__init__()
        self.l = nn.Linear(real_dimension,d_model)
    def forward(self,x):
        return self.l(x)

class Conv2dLayers(nn.Module):
    def __init__(self):
        super(Conv2dLayers,self).__init__()
        self.conv2d = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,padding=1)
    def forward(self,x):
        x = x.unsqueeze(1)
        x_1 = self.conv2d(x)
        x_1=x_1.view(x_1.shape[0],-1,x_1.shape[3]).transpose(1,2)
        return  x_1
class GatedCNN(nn.Module):
    def __init__(self):
        super(GatedCNN,self).__init__()
        self.conv2d = nn.Conv2d(in_channels=1,out_channels=5,kernel_size=3,padding=1)
        self.b_0 = nn.Parameter(torch.randn(1,5,1,1))
        self.gated_conv2d = nn.Conv2d(in_channels=1,out_channels=5,kernel_size=3,padding=1)
        self.c_0 = nn.Parameter(torch.randn(1,5,1,1))
    def forward(self, x):
        x = x.unsqueeze(1)
        conv_x = self.conv2d(x)                               #Batch_size*in_channel*feature_dim*seq_len
        conv_x = conv_x + self.b_0.repeat(1,1,x.shape[2],x.shape[3]) #1*5*feature*seq_len
        conv_gate = self.gated_conv2d(x)
        conv_gate = conv_gate+self.c_0.repeat(1,1,x.shape[2],x.shape[3])
        output = conv_x * torch.sigmoid(conv_gate)
        output = output.view(output.shape[0],-1,output.shape[3])
        return output.transpose(1,2)
class GatedMechanism(nn.Module):
    def __init__(self,d_model):
        super(GatedMechanism,self).__init__()
        #update gate
        self.w_u = nn.Parameter(torch.randn(d_model,d_model)) #128*128
        self.v_u = nn.Parameter(torch.randn(d_model,d_model)) #128*128
        self.b_u = nn.Parameter(torch.randn(d_model)) #128
        #reset gate
        self.w_r = nn.Parameter(torch.randn(d_model,d_model))
        self.v_r = nn.Parameter(torch.randn(d_model,d_model))
        self.b_r = nn.Parameter(torch.randn(d_model))
    def forward(self,origin_data,hidden_data):
        #update gate
        origin_data = origin_data.transpose(1,2)
        update_hidden = torch.matmul(hidden_data,self.w_u)#8*129*128
        update_origin = torch.matmul(origin_data,self.v_u) #8*129*128
        update_gate = torch.sigmoid( update_hidden + update_origin + self.b_u)
        #reset gate
        reset_hidden = torch.matmul(hidden_data,self.w_r)
        reset_origin = torch.matmul(origin_data,self.v_r)
        reset_gate = torch.sigmoid( reset_hidden + reset_origin + self.b_r)
        output = update_gate * hidden_data + reset_gate * origin_data
        return output
class OutputLinears(nn.Module):
    def __init__(self,d_model,out_dimension):
        super(OutputLinears,self).__init__()
        self.output = nn.Linear(d_model,out_dimension)
    def forward(self, x):
        x = torch.sigmoid(self.output(x).squeeze(2))
        return x
def subsequent_mask(size):  #构造了一个上三角矩阵，对角线以下（包含对角线）元素为0,tgt_mask，size是输出翻译句子的长度
    attn_shape = (1,size,size)
    subsequent_mask = np.triu(np.ones(attn_shape),k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
def make_std_mask(batch_size,seq_size):
    '''create a mask to hide padding and future words'''
    tgt_mask = (torch.randn(batch_size,seq_size)!=0).unsqueeze(1)  #batch_size * 1 * seq #编码器结构
    #tgt_mask = tgt_mask & subsequent_mask(seq_size) #这里返回的subsequent_mask就是(1,24,24) 最后得到的就是(48,24,24)
    return tgt_mask
####################################################
class Model(nn.Module):
    def __init__(self,real_dimension,N=2,d_model=128,d_ff=512,h=4,out_dim=1,dropout=0.1):
        super(Model,self).__init__()
        print("编码器层数:",N)
        self.conv2d = Conv2dLayers()
        self.GatedCNN=GatedCNN()
        self.gate_machism = GatedMechanism(real_dimension)
        self.input_linear = InputLinears(real_dimension,d_model)
        self.embedding = PositionalEncoding(d_model,dropout)
        self.attn = MultiHeadAttention(h,d_model)
        self.ff = PositionwiseFeedForward(d_model,d_ff,dropout)
        self.FinalDecoder = Decoder(DecoderLayer(d_model,self.attn,self.ff,dropout),N)
        self.output_layer = OutputLinears(d_model,out_dim)
    def forward(self,tgt,tgt_mask):                  #tgt bs*feature*seq
        Conv2d_hidden = self.conv2d(tgt)             #bs*seq*feature  feature=14
        tgt = self.gate_machism(tgt, Conv2d_hidden)  #bs*seq*feature
        # tgt = self.GatedCNN(tgt)
        inputlayer_hidden = self.input_linear(tgt)   #
        embedding_hidden = self.embedding(inputlayer_hidden)
        Transformer_hidden = self.FinalDecoder(embedding_hidden,tgt_mask)
        x = self.output_layer(Transformer_hidden)
        return x