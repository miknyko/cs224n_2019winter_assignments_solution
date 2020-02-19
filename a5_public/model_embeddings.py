#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""
import torch
import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        pad_token_idx = vocab.char2id['<pad>']
        self.cnn = CNN(e_word=embed_size)
        self.highway = Highway(dim=embed_size)
        self.dropout = nn.Dropout(0.3)
        self.embeddings = nn.Embedding(len(vocab.char2id),embedding_dim=50,padding_idx=pad_token_idx)   # 这里一定注意embedding层的大小，不能直接引用len(vocab)
        self.embed_size = embed_size
    

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code
        
        x_data = []                          # 创建数据空列表
        for X_padded in input:            # 将整个训练数据切分为(batch_size,max_word_length)大小的minibatch进行批处理
            X_emb = self.embeddings(X_padded)     # 进行embedding,(batch_size,max_word_length,e_char)
            x = self.cnn(X_emb)                   # 进行CONV-1D,(batch_size,e_word) 
            x = self.highway(x)                    # 输入highway,(batch_size,e_word)
            x = self.dropout(x)                    # dropout,(barch_size,e_word)                             
            x_data.append(x)                      # 将结果加至列表

        output = torch.stack(x_data)              #将数据格式还原为(sentence_length,batch_size,e_word)
        
        return output
        


        # （第一次错误代码）
        # char_emb = self.embeddings(input)
        # sentence_length,batch_size,max_word_length = input.size()
        # x = char_emb.view(-1,char_emb.size(2),char_emb.size(3)) # 将前两维压缩至一维，以便后面的cnn正传播
        # x = self.cnn(x)
        # x = self.highway(x)  # 大小(batch,word_em)
        # x = self.dropout(x)
        # output = x.view(sentence_length,batch_size,-1) # 

        # return output



        ### YOUR CODE HERE for part 1j



        ### END YOUR CODE

