#!/usr/bin/env python
# coding: utf-8

model_name='Img_to_value_model_4 (use h)'

import torch
import torch.nn as nn
from My_enc_dec_convLSTM_model import Encoder

class my_img_predict_value_model(nn.Module):
    def __init__(self,input_dim):
        super(my_img_predict_value_model, self).__init__()
        self.input_dim=input_dim
        
        self.conv1=nn.Conv2d(in_channels=self.input_dim,
                             out_channels=32,
                             kernel_size=(3,3),
                             stride=1,
                             padding=1)
        self.conv2=nn.Conv2d(in_channels=32,
                             out_channels=32,
                             kernel_size=(4,4),
                             stride=2,
                             padding=1)
        self.conv3=nn.Conv2d(in_channels=32,
                             out_channels=64,
                             kernel_size=(3,3),
                             stride=1,
                             padding=1)
        self.conv4=nn.Conv2d(in_channels=64,
                             out_channels=64,
                             kernel_size=(4,4),
                             stride=2,
                             padding=1)
        '''self.convLSTM=My_enc_dec_convLSTM_model(input_dim=64,
                                                hidden_dim=64, 
                                                kernel_size=(3,3), 
                                                bias=True,
                                                dec_seq_len=self.dec_seq_len,
                                                num_layers=1)'''
        self.convLSTM=Encoder(input_dim=64, 
                             hidden_dim=64,
                             kernel_size=(3,3),
                             bias=True)
        self.conv5=nn.Conv2d(in_channels=64,
                             out_channels=32,
                             kernel_size=(3,3),
                             stride=1,
                             padding=1)
        self.conv6=nn.Conv2d(in_channels=32,
                             out_channels=16,
                             kernel_size=(3,3),
                             stride=1,
                             padding=1)
        self.conv7=nn.Conv2d(in_channels=16,
                             out_channels=1,
                             kernel_size=(3,3),
                             stride=1,
                             padding=1)
        self.conv0=nn.Conv2d(in_channels=64,
                             out_channels=1,
                             kernel_size=(3,3),
                             stride=1,
                             padding=1)
        self.conv32_1=nn.Conv2d(in_channels=32,
                             out_channels=1,
                             kernel_size=(3,3),
                             stride=1,
                             padding=1)

        self.pool=nn.MaxPool2d(2,2)
        self.pool0=nn.MaxPool2d(4,4)

        self.activation=nn.ReLU()
        
        self.bn_input=nn.BatchNorm2d(num_features=self.input_dim)
        self.bn16=nn.BatchNorm2d(num_features=16)
        self.bn32=nn.BatchNorm2d(num_features=32)
        self.bn64=nn.BatchNorm2d(num_features=64)

        self.flatten=nn.Flatten()
        self.fc=nn.Linear(8*8,1)

    def forward(self,input_):
        x_list=[]
        for t in range(input_.shape[-4]):
            
            x=self.conv1(input_[:,t])
            x=self.conv2(x)
            x=self.conv3(x)
            x=self.conv4(x)
            x=self.bn64(x)

            x_list.append(x)
        x=torch.stack(x_list,dim=1)
        
        #x=nn.BatchNorm2d(x.shape[-3])(x)
        h_out, _ = self.convLSTM(x) #h_out:(batch,channels,h,w)
        
        out=self.conv5(h_out)
        out=self.pool(out)
        #out=self.activation(out)
        out=self.bn32(out)

        out=self.conv6(out)
        out=self.pool(out)
        out=self.bn16(out)

        out=self.conv7(out)
        out=self.flatten(out)
        out=self.fc(out)

        return out
