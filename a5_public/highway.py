#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch

class Highway(torch.nn.Module):
    def __init__(self,dim):
        """
        初始化一个highway模块
        @param dim(int) : feature map大小，在此网络中输出和输入具有同样大小
        """

        super(Highway,self).__init__()
        self.shortcut = torch.nn.Linear(dim,dim)
        self.mainroad = torch.nn.Linear(dim,dim)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        """
        highway正传播
        @param x 卷积网络输出x_conv,张量,维度为(batch_size,dim)
        @returns x_highway 张量,维度为(batch_size,dim)
        """
        x_proj = self.shortcut(x)
        x_proj = self.relu(x_proj)
        x_gate = self.mainroad(x)
        x_gate = self.sigmoid(x_gate)
        x_highway = torch.mul(x_proj,x_gate) + torch.mul((1 - x_gate),x)

        return x_highway

### END YOUR CODE 

if __name__ == "__main__":
    print('[INFO] 测试Highway模块...')
    batch_size = 3
    dim = 5
    net = Highway(dim)
    x = torch.randn(batch_size,dim)
    output = net.forward(x)
    
    assert (output.size() == (batch_size,dim)),f"输出张量大小应该为{(batch_size,dim)},而得到的张量大小为{output.size()}"
    print('[INFO] 测试通过！')