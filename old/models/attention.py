# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import data.dict as dict

class global_attention(nn.Module):

    def __init__(self, hidden_size, activation=None):
        super(global_attention, self).__init__()
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(2*hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)  # batch*time
        self.tanh = nn.Tanh()
        self.activation = activation

    def forward(self, x, context):
        """
        execute the attention

        :param x: the output of a timestep rnn, shape [batch, hid*num_dirc] (often used in decoder,
                so the num_dirc is always 1)
        :param context: context from encoder, shape [batch, maxlen, hidden*num_dirc]
        :return: output: result calculated by attention, shape [batch, hid*num_dir]
                weights: the attention weight this timestep to all timesteps in encoder, shape[batch, time]
        """

        gamma_h = self.linear_in(x).unsqueeze(2)    # batch * size * 1
        if self.activation == 'tanh':  # TODO: activation is still None now
            gamma_h = self.tanh(gamma_h)
        weights = torch.bmm(context, gamma_h).squeeze(2)  # [batch, maxlen, hid] dot [batch, hid, 1] # batch * time
        weights = self.softmax(weights)   # batch * time  # Question: 为什么这里debug不停啊……
        # calculate weight over all context
        c_t = torch.bmm(weights.unsqueeze(1), context).squeeze(1)  # [batch, 1, time] dot [batch, maxlen, hid] # batch * size
        output = self.tanh(self.linear_out(torch.cat([c_t, x], 1)))
        return output, weights
