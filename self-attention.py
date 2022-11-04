'''
自注意力机制
'''
import math
import torch
import torch.nn as nn

# 位置编码
class PositionalEncoding(nn.Module):
    '''
    以下是基于正弦函数和余弦函数的固定位置编码
    假定输入表示X(属于n x d维实数空间)包含一个序列中n个词元的d维嵌入表示
    位置编码使用相同形状的位置嵌入矩阵P(属于n x d维实数空间)输出X+P
    '''
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1,1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32)/num_hiddens)
        
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X+self.P[:,:X.shape[1], :].to(X.device)
        return self.dropout(X)

# 注意力机制
def sequence_mask(X, valid_len, value=0):
    '''
    序列中添加掩码, 屏蔽不相关的项
    '''
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X
def masked_softmax(X, valid_lens):
    '''
    通过valid_lens选择有无掩码执行softmax
    '''
    if valid_lens is None: # 无掩码
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1]) # 重复张量的元素,[1,2,3]->[1,1,2,2,3,3]
        else:
            valid_lens = valid_lens.reshape(-1) # 存疑
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6) # 最后一轴上被掩码的元素用非常大的赋值代替，使得softmax输出为0
        return nn.functional.softmax(X.reshape(shape), dim=-1)
class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        attention = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d) # torch.bmm计算张量的矩阵乘法
        self.attention_weights = masked_softmax(attention, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

def transpose_qkv(X, num_heads):
    '''
    适应多头注意力的计算需要变换形状
    输入为(batch_size,键值对数目,num_hiddens)
    输出为(batch_size,键值对数目,num_heads, num_hiddens/num_heads)
    '''
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # 输出X的形状:(batch_size,num_heads,键值对数目,num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # 最终输出的形状:(batch_size*num_heads,键值对数目,num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    '''
    逆转transpose_qkv的操作
    '''
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)