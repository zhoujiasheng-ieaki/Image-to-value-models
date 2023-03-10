#!/usr/bin/env python
# coding: utf-8

# ConvLSTMCell from https://github.com/ndrplz/ConvLSTM_pytorch

# # ConvLSTMCell でゼロから組立てる

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))



class Encoder(nn.Module):
    '''
    input:
    x_seq: 5D tensor ( batch, seq, channel, height, width )
        
    output:
    h_out, c_out: (final out) 4D tensor ( batch, hidden_dim, height, width ) 
    '''
    def __init__(self, input_dim, hidden_dim, kernel_size, bias): #seq_len = seq_lens of input x_data! 入力時系列に依存する
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.bias = bias
                
        self.cell=ConvLSTMCell(self.input_dim, self.hidden_dim, self.kernel_size, self.bias)

    
    def forward(self,x_seq):
        batch, seq, channel, height, width =x_seq.shape
        h, c =self.cell.init_hidden(batch,(height,width))
        for t in range(seq):
            h, c =self.cell(x_seq[:,t,:,:,:],(h, c))
        
        return h, c
        
        
class Decoder(nn.Module):
    '''
    input:
    x: first one only 4D tensor ( batch, channel, height, width )
    h_in, c_in: (first in) 4D tensor ( batch, hidden_dim, height, width ) 
    
    output:
    out: 5D tensor ( batch, time_seq, in_channel(in_dim), height, width )
    h_out, c_out: (final out) 4D tensor ( batch, hidden_dim, height, width ) 
    '''
    def __init__(self, input_dim, hidden_dim, kernel_size, bias, seq_len, num_layers=1):
        super(Decoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.bias = bias
        self.seq_len=seq_len
        self.num_layers=num_layers
        
        self.cell=ConvLSTMCell(self.input_dim, self.hidden_dim, self.kernel_size, self.bias)
        self.conv2d=nn.Conv2d(in_channels=self.hidden_dim,
                              out_channels=self.input_dim,
                              kernel_size=(1,1),
                              stride=1) # (N,C,H,W)
        #self.bn=nn.BatchNorm2d(num_features=1,affine =False,track_running_stats=False)
        
    def forward(self,x,state): #　xは最初の１個だけ（seqなし）（batch, channel, height, width）
        h_in, c_in = state
                
        h_out, c_out =self.cell(x,(h_in, c_in))
        out_temp=(self.conv2d(h_out))
        out=out_temp[:,None,:,:,:] #---> (batch, time_seq, channel, height, width) 　yのshapeと同じに
        
        if self.seq_len-1 > 0: #seq_len > 1
            for _ in range(self.seq_len-1): #最初のXを入力とするCellを除き、残りのCell部分を繰り返す
                h_out, c_out =self.cell(out_temp,(h_out, c_out)) # 直前回のout_tempを外部からの
                #out_temp=self.bn(self.conv2d(h_out))
                out_temp=self.conv2d(h_out)
                out=torch.cat([out,out_temp[:,None,:,:,:]],dim=1) # seqで結合　listの代わりに
        
        if self.num_layers > 1:
            for layer_index in range(self.num_layers-1): #stacked convLSTM
                layer_seq_input=out #(batch, time_seq, channel, height, width)
                
                h_out, c_out =self.cell(layer_seq_input[:,0,:,:,:],(h_in, c_in))
                out_temp=self.conv2d(h_out)
                out=out_temp[:,None,:,:,:] #---> (batch, time_seq, channel, height, width) 　yのshapeと同じに
                
                for seq_index in range(self.seq_len-1): #最初のh_in,c_inを入力とするCellを除き、残りのCell部分を繰り返す
                    h_out, c_out =self.cell(layer_seq_input[:,seq_index+1,:,:,:],(h_out, c_out))
                    out_temp=self.conv2d(h_out)
                    out=torch.cat([out,out_temp[:,None,:,:,:]],dim=1) # seqで結合　listの代わりに
        
        return out, h_out, c_out
    
    

class My_enc_dec_convLSTM_model(nn.Module):
    '''
    My(zjs) original Encoder-Decoder model
    
    input:
    x_seq: 5D tensor ( batch, seq, channel, height, width )
    
    output:
    out: 5D tensor ( batch, time_seq, in_channel(in_dim), height, width )
    h_out, c_out: (final out) 4D tensor ( batch, hidden_dim, height, width )
    
    '''
    
    def __init__(self, input_dim, hidden_dim, kernel_size, bias, dec_seq_len,num_layers=1):
        super(My_enc_dec_convLSTM_model, self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.kernel_size=kernel_size
        self.bias=bias
        self.dec_seq_len=dec_seq_len
        self.num_layers=num_layers
        
        self.encoder=Encoder(input_dim=self.input_dim, 
                             hidden_dim=self.hidden_dim,
                             kernel_size=self.kernel_size,
                             bias=self.bias)
        self.decoder=Decoder(input_dim=self.input_dim,
                             hidden_dim=self.hidden_dim,
                             kernel_size=self.kernel_size, 
                             bias=self.bias, 
                             seq_len=self.dec_seq_len,
                             num_layers=self.num_layers)
    
    def forward(self,x_seq):
        h_middle,c_middle = self.encoder(x_seq)
        out, h_out, c_out = self.decoder(x_seq[:,-1,:,:,:],(h_middle,c_middle))
        
        return out
