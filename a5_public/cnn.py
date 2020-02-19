#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch

class CNN(torch.nn.Module):
    """
    卷积网络
    """
    def __init__(self,e_word,k=5):
        """
        卷积网络初始化
        @param e_word(int):输出通道数量，也即是最后word embedding的维数
        @param e_char(int):输入通道数量，也即是开始的char embedding的维数
        @param k(int):卷积核数量，默认k=5
        """
        super(CNN,self).__init__()
        self.e_char = 50
        self.conv1d = torch.nn.Conv1d(self.e_char,e_word,k)  # 这里的第一个参数in_channel实际上就是张量倒数第二维，即char_emb维数
        self.relu = torch.nn.ReLU()
        self.k = k

    def forward(self,x):
        """
        CNN正向传播
        @param x (tensor,size:(batch_size,max_word_length,e_char)): 字母embedding的输出
        @returns x_convout (tensor,size:(batch_size,e_word)):预备输入highway模块
        """
        batch_size,max_word_length,e_char = x.size()
        x = x.permute(0,2,1) # 将最后两维置换，以配合Pytorch的conv1d
        x = self.conv1d(x)
        x = self.relu(x)
        x = torch.nn.MaxPool1d(max_word_length - self.k + 1)(x) # 沿着最后一维进行maxpool
        x_convout = torch.squeeze(x,-1) # 这里一定要指定最后一维squeeze，否则当出现batch size为1的数据时，也会被squeeze
        
        return x_convout


if __name__ == "__main__":
    print('[INFO] 测试CNN模块...')
    batch_size = 10
    # max_sentence_length = 5
    max_word_length = 8
    e_char = 50
    e_word = 9
    net = CNN(e_word,k=5)
    x = torch.randn(batch_size,max_word_length,e_char)
    output = net.forward(x)

    assert (output.size() == (batch_size,e_word)),f"[WARNING] 测试结果为{output.size()},[WARNING] 应该为{batch_size,e_word}"
    print("[INFO] 测试通过！")
### END YOUR CODE

