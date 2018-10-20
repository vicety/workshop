# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import data.dict as dict
import models

import numpy as np


# 用于计算一个时步的LSTM，相当于是rnn中的cell定义
class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    # input (batch, embed)
    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)  # why dimension 0?
        c_1 = torch.stack(c_1)

        # input is the outer stacked-layer, (h1, c1) contains all layers
        return input, (h_1, c_1)


class rnn_encoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None, tgt_embedding=None):
        super(rnn_encoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        if config.k_max != 0:
            self.rnn = nn.LSTM(input_size=config.emb_size * (config['k_max'] * 2 + 1), hidden_size=config.encoder_hidden_size, num_layers=config.num_layers, dropout=config.dropout, bidirectional=config.bidirec)
        else:
            self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.encoder_hidden_size, num_layers=config.num_layers, dropout=config.dropout, bidirectional=config.bidirec)
        
        self.tgt_embedding = tgt_embedding
        self.config = config

    def forward(self, input, lengths):
        '''
        :param input: src [maxlen, batch]
        :param lengths: list([batch])
        :return: output, (h, c) 定义见下
        embs维度[maxlen, batch, embed]
        output [maxlen, batch, hidden * num_directions] encoder的输出
        h, c [num_layer * num_dir, batch, hidden_sz]
        如果是双向rnn，那么将h, c维度转换为[num_layer, batch, 2*hidden]
        '''
        input_emb = self.embedding(input)
        if not self.config.k_max:
            input_feature = input_emb
        else:    
            tgt_emb = self.tgt_embedding.weight.detach()
            input_emb_score = torch.matmul(input_emb, tgt_emb.t())
            k_max_index = input_emb_score.topk(self.config['k_max'], dim=2)[1]
            doc, batch, k_max = k_max_index.shape
            '''
            res = []
            for i in range(doc):
                for j in range(batch):
                    for k in range(k_max):
                        res.append(tgt_emb[k_max_index[i][j][k]])
            k_max_embed = torch.cat(res, dim=0).view(doc, batch, k_max, -1)
            '''
            k_max_embed = self.tgt_embedding(k_max_index)
            input_emb = input_emb.unsqueeze(2)
            minus = (k_max_embed - input_emb).view(doc, batch, -1)
            mul = (k_max_embed * input_emb).view(doc, batch, -1)
            input_feature = torch.cat([input_emb.squeeze(), mul, minus], dim=2)
        

        embs = pack(input_feature, lengths)  # 尽管有些难接受，embedding不一定要单词维度在最后……
        outputs, (h, c) = self.rnn(embs)
        outputs = unpack(outputs)[0]  # unpack returns (padded_seq, it's length)
        if not self.config.bidirec:
            return outputs, (h, c)  # h, c -> [num_layer, batch, hidden]
        else:
            batch_size = h.size(1)
            # view only make-sense to a contiguous Tensor, however, transpose makes it not contiguous
            h = h.transpose(0, 1).contiguous().view(batch_size, -1, 2 * self.config.encoder_hidden_size)
            c = c.transpose(0, 1).contiguous().view(batch_size, -1, 2 * self.config.encoder_hidden_size)

            state = (h.transpose(0, 1), c.transpose(0, 1))  # h, c -> [num_layer, batch, 2*hidden]
            return outputs, state


class gated_rnn_encoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None):
        super(gated_rnn_encoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.encoder_hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout)
        self.gated = nn.Sequential(nn.Linear(config.encoder_hidden_size, 1), nn.Sigmoid())

    def forward(self, input, lengths):
        embs = pack(self.embedding(input), lengths)
        outputs, state = self.rnn(embs)
        outputs = unpack(outputs)[0]
        p = self.gated(outputs)
        outputs = outputs * p
        return outputs, state


class rnn_decoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None, score_fn=None):
        super(rnn_decoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        self.rnn = StackedLSTM(input_size=config.emb_size, hidden_size=config.decoder_hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout)

        self.score_fn = score_fn
        if self.score_fn.startswith('general'):
            self.linear = nn.Linear(config.decoder_hidden_size, config.emb_size)
        elif score_fn.startswith('concat'):
            self.linear_query = nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size)
            self.linear_weight = nn.Linear(config.emb_size, config.decoder_hidden_size)
            self.linear_v = nn.Linear(config.decoder_hidden_size, 1)
        elif score_fn == 'hinge_margin_loss':
            self.toEmbLinear = nn.Linear(config.decoder_hidden_size, config.emb_size)
            if config.global_emb:
                self.linear = nn.Linear(config.decoder_hidden_size, vocab_size)  # for global_embedding
        elif score_fn == 'hybrid':
            # Question： 有空搞清楚这里config的形式
            # 转换到Emedding空间
            self.toEmbLinear = nn.Linear(config.decoder_hidden_size, config.emb_size)
            self.toCatLinear = nn.Linear(config['emb_size'], vocab_size)
            self.grl = models.GradReverse(config['GRL_fraction'])
            self.activation = nn.Tanh()
            if config.global_emb:
                self.linear = nn.Linear(config.decoder_hidden_size, vocab_size)
        elif score_fn == 'hubness':
            self.emb_to_mid1 = nn.Linear(config.emb_size, config.decoder_hidden_size)
            self.mid1_to_mid2 = nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size)
            self.activation = nn.ReLU()
            if config.global_emb:
                self.linear = nn.Linear(config.decoder_hidden_size, vocab_size)
        elif score_fn == 'disc':
            self.toEmbLinear = nn.Linear(config.decoder_hidden_size, config.emb_size)
            self.bin_classify = nn.Linear(config.decoder_hidden_size, 2)
            self.grl = models.GradReverse(config['GRL_fraction'])
            self.activation = nn.Tanh()
            if config.global_emb:
                self.linear = nn.Linear(config.decoder_hidden_size, vocab_size)
            # TODO
        # score_fn == softmax的情形
        elif not self.score_fn.startswith('dot'):
            self.linear = nn.Linear(config.decoder_hidden_size, vocab_size)

        if hasattr(config, 'att_act'):
            activation = config.att_act
            print('use attention activation %s' % activation)
        else:
            activation = None

        self.attention = models.global_attention(config.decoder_hidden_size, activation)
        self.hidden_size = config.decoder_hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.config = config

        if self.config.global_emb:
            self.gated1 = nn.Linear(config.emb_size, config.emb_size)  # 论文公式(13）的两个W矩阵
            self.gated2 = nn.Linear(config.emb_size, config.emb_size)

    # contexts [batch, maxlen, hidden*num_dirc]  # inputs [maxlen, batch]
    def forward(self, inputs, init_state, contexts):
        '''
        :param inputs: trg去除EOS [maxlen, batch]（因为EOS是希望rnn最后一个时步预测出来的，而不是我们输入给rnn的）
        :param init_state: h, c [num_layer * num_dir, batch, hidden_sz]
        :param contexts: [batch, maxlen, hidden * num_directions] 来自encoder的输出
        :return: outputs [time, batch, hidden*num_dir] 所有时步的attention output
                state (h, c) [num_layer * num_dir, batch, hidden_sz] 最后一个时步的RNN状态
        embs [maxlen, batch, embed_size]
        decoder需要一个个时步计算，因为需要中间过程做attention，因此用的不是nn.LSTM，而是自己定义的StackedLSTM
        每个时步：
            emb [1, batch, embed_size]
            output [batch, hidden * num_directions]
            state = (h, c) [num_layer * num_dir, batch, hidden_sz]
            过attention，output [batch, dec_hidden], weights [batch, time]

        当使用global_embedding时，第一次使用普通的embedding，其余所有时步都使用论文中的embedding
        论文 https://arxiv.org/pdf/1806.04822.pdf 公式(11) (12) (13)
        '''
        # TODO: make a better attention
        if not self.config.global_emb:
            embs = self.embedding(inputs)  # embs [maxlen, batch, embed_size]
            outputs, state, attns = [], init_state, []
            # split的第一个参数是每份的大小，默认dim=0
            for emb in embs.split(1):  # emb [1, batch, embed_size] # 可以看出喂给rnn的不是上一预测状态，而是target句
                                                                # 而在evaluate的时候是输入上一个预测
                # 这里只喂了一个时步
                # 输入(1, batch, embed_size) 1代表1个时步
                output, state = self.rnn(emb.squeeze(0), state)  # output [batch, hidden * num_directions]
                                          # h, c [num_layer * num_dir, batch, hidden_sz]
                # output [batch, dec_hidden], weights [batch, time]
                # 这里的attention是使output结合encoder信息，而不是decoder的input
                output, attn_weights = self.attention(output, contexts)
                output = self.dropout(output)
                outputs += [output]
                attns += [attn_weights]
            outputs = torch.stack(outputs)
            attns = torch.stack(attns)  # 是空的，并没有用上
            return outputs, state
        else:
            outputs, state, attns = [], init_state, []
            embs = self.embedding(inputs).split(1)  # [maxlen(list维), batch, embed_size]
            max_time_step = len(embs)
            emb = embs[0]
            output, state = self.rnn(emb.squeeze(0), state)
            output, attn_weights = self.attention(output, contexts)  # output [batch, dec_hidden], weights [batch, time]
            output = self.dropout(output)
            soft_score = F.softmax(self.linear(output), dim=1)  # [batch, trg_vocab]
            outputs += [output]
            attns += [attn_weights]

            batch_size = soft_score.size(0)
            a, b = self.embedding.weight.size()

            for i in range(max_time_step-1):
                # embedding weight 就是embedding矩阵
                # [batch, 1, trg_vocab] dot [batch, trg_vocab, embed] -> [batch, 1, embed] the average(e) in 论文(11)
                emb1 = torch.bmm(soft_score.unsqueeze(1), self.embedding.weight.expand((batch_size, a, b)))
                emb2 = embs[i+1]  # 论文中的the label with highest probability, 在这里就直接是正确答案的embed了……
                                  # 但在evaluate的时候是
                # gamma [batch, embed]
                gamma = F.sigmoid(self.gated1(emb1.squeeze())+self.gated2(emb2.squeeze()))  # 默认squeeze去除所有1维
                emb = gamma * emb1.squeeze() + (1 - gamma) * emb2.squeeze()  # 论文公式（12）
                output, state = self.rnn(emb, state)
                output, attn_weights = self.attention(output, contexts)  # output [batch, dec_hidden], weights [batch, time]
                output = self.dropout(output)
                soft_score = F.softmax(self.linear(output), dim=1)
                outputs += [output]
                attns += [attn_weights]
            outputs = torch.stack(outputs)
            attns = torch.stack(attns)
            return outputs, state

    # hiddens[time * batch, dec_hidden]
    def compute_score(self, hiddens):
        '''
        :param hiddens: [time * batch, dec_hidden] 来自decoder
        :return:
        各种不同的score计算方式，默认(when score_fn == None)是一个linear
        '''
        if self.score_fn.startswith('general'):
            if self.score_fn.endswith('not'):
                scores = torch.matmul(self.linear(hiddens), Variable(self.embedding.weight.t().data))
            else:
                scores = torch.matmul(self.linear(hiddens), self.embedding.weight.t())
        elif self.score_fn.startswith('concat'):
            if self.score_fn.endswith('not'):
                scores = self.linear_v(torch.tanh(self.linear_query(hiddens).unsqueeze(1) + \
                                      self.linear_weight(Variable(self.embedding.weight.data)).unsqueeze(0))).squeeze(2)
            else:
                scores = self.linear_v(torch.tanh(self.linear_query(hiddens).unsqueeze(1) + \
                                      self.linear_weight(self.embedding.weight).unsqueeze(0))).squeeze(2)
        elif self.score_fn == 'hinge_margin_loss':
            # [time*batch, emb]
            toEmb = self.toEmbLinear(hiddens)
            # restrict L2 norm to be 1
            toEmb = toEmb.div(torch.norm(toEmb, 2, 1).unsqueeze(1))
            # _ =  self.embedding.weight.t().detach() # false
            # __ = self.embedding.weight.t().data  # false
            embedding_cp = self.embedding.weight.detach()
            embedding_cp = embedding_cp.div(torch.norm(embedding_cp, 2, 1).unsqueeze(1))
            scores = torch.matmul(toEmb, embedding_cp.t())
        elif self.score_fn == 'hybrid':
            # HINT: 由于这里只是为了测试hybrid，因此return的score不兼容其他score_fn
            emb_space_vector = self.toEmbLinear(hiddens)  # 实际上还没有做softmax
            # emb_score_vector = self.activation(emb_space_vector)

            emb_space_vector_normed = models.utils.l2norm(emb_space_vector)
            embedding_cp_normed = models.utils.l2norm(self.embedding.weight.detach())
            if self.training:
                scores = {}
                scores['softmax'] = self.toCatLinear(self.grl(emb_space_vector))
                scores['margin'] = torch.matmul(emb_space_vector_normed, embedding_cp_normed.t())
            else:
                scores = torch.matmul(emb_space_vector_normed, embedding_cp_normed.t())
                '''   
                import time
                print(scores[:5])
                time.sleep(10)
                '''
        elif self.score_fn == 'hubness':
            mid1 = self.emb_to_mid1(self.embedding.weight.detach())
            mid1 = self.activation(mid1)  # relu
            mid2 = self.mid1_to_mid2(mid1)
            mid2 = self.activation(mid2)
            mid2_normed = models.utils.l2norm(mid2)
            hiddens_normed = models.utils.l2norm(hiddens)
            scores = torch.matmul(hiddens_normed, mid2_normed.t())
        elif self.score_fn == 'disc':
            # hint: filter留到score去做
            emb_space_vector = self.toEmbLinear(hiddens)  # 实际上还没有做softmax
            # emb_score_vector = self.activation(emb_space_vector)

            emb_space_vector_normed = models.utils.l2norm(emb_space_vector)
            embedding_cp_normed = models.utils.l2norm(self.embedding.weight.detach())
            if self.training:
                scores = {}
                scores['softmax'] = self.bin_classify(self.grl(hiddens))
                scores['margin'] = torch.matmul(emb_space_vector_normed, embedding_cp_normed.t())
            else:
                scores = torch.matmul(emb_space_vector_normed, embedding_cp_normed.t())
        elif self.score_fn.startswith('dot'):
            if self.score_fn.endswith('not'):
                scores = torch.matmul(hiddens, Variable(self.embedding.weight.t().data))
            else:
                scores = torch.matmul(hiddens, self.embedding.weight.t())
        else:  # when score_fn == None
            scores = self.linear(hiddens)
        return scores

    def sample(self, input, init_state, contexts):
        '''
        :param input: [Variable of size [batch]] 通常是bos标签
        :param init_state: (h, c) -> [num_layer, batch, dir * hidden] 来自encoder
        :param contexts: [batch, maxlen, hidden*num_dirc] 来自encoder
        :return:
        1.对每个timestep sample_one，输入上一个时步的预测id(shape=[batch])
            soft_score [batch, vocab_size]
            state (h, c) -> [num_layer, batch, dir * hidden] 来自rnn（第一个时步是encoder的，其他是decoder rnn的）
            contexts: [batch, maxlen, hidden*num_dirc] 来自encoder
            mask： 用于过滤已经选择的标签
            返回  output [batch, vocab_trg] 其实是score
                attn_weigths the attention weight this timestep to all timesteps in encoder, shape[batch, time]
        2.对score做softmax得soft_score
        3.做概率最优的预测，predicted shape=[batch]
        4.predicted加入mask
        5.返回sample_ids shape[time, batch]
            outputs [time, batch, vocab_trg]
            attn_weights shape[time(dec), batch, time(enc)]
        '''
        inputs, outputs, sample_ids, state = [], [], [], init_state
        attns = []
        inputs += input
        max_time_step = self.config.max_tgt_len
        soft_score = None
        mask = None
        for i in range(max_time_step):
            # output（这里其实是score） -> 过attention后再计算的score [batch, vocab_trg], state(单步过rnn,[num_layer, batch, dir*hidden]), attn_weigths
            # HINT： 也就是说sample_one的平均
            output, state, attn_weights = self.sample_one(inputs[i], soft_score, state, contexts, mask)
            if self.config.global_emb:
                soft_score = F.softmax(output)
            predicted = output.max(1)[1]  # 第一个1指dim, 返回(值, index)
            inputs += [predicted]  # [以BOS开头，所有生成的都往里放]  # 这里确实是贯彻了论文的取最高概率
            sample_ids += [predicted]  # hint: [应该就是inputs除掉BOS],从这里可以看出bos是不含在预测句中的
            outputs += [output]
            attns += [attn_weights]
            if self.config.mask:
                if mask is None:  # 第一次
                    mask = predicted.unsqueeze(1).long()
                else:
                    mask = torch.cat((mask, predicted.unsqueeze(1)), 1)  # 用于更新mask（所有已经选择的label）

        sample_ids = torch.stack(sample_ids)
        attns = torch.stack(attns)  # attention_weight over all timestep
        return sample_ids, (outputs, attns)

    # from beam_sample
    # Run one step. # inp[80], soft_score -> initially None  decState->[layer, batch*bean_sz, dir*hidden]
    # contexts -> [batch * beam_sz, len, hid*dirc], mask initially None

    # from sample
    # state -> (h, c) -> [num_layer, batch, dir * hidden]  # input [BOS]
    # contexts [batch, maxlen, hidden*num_dirc]
    def sample_one(self, input, soft_score, state, contexts, mask):
        '''
        :param input: [beam_sz*batch, embed](from beam_sample) [batch, embed](from sample 只选最高概率)
        :param soft_score: [batch, vocab]
        :param state:
        :param contexts:
        :param mask: 用于过滤已经选择的标签
        :return: output [batch, vocab_trg] 在target_vocab上的score
            attn_weigths the attention weight this timestep to all timesteps in encoder, shape[batch, time]
        1.按论文公式综合计算emb1, emb2
            Arg emb1: softscore对embedding的加权平均
            Arg emb2： 最优概率input的embedding
        2.过rnn得output state
            Arg output [batch, hidden * num_directions]
            Arg state = h, c [num_layer * num_dir, batch, hidden_sz]
        3.过attention的output attn_weights
            Arg output: result calculated by attention, shape [batch, hid*num_dir]
            Arg attn_weights: the attention weight this timestep to all timesteps in encoder, shape[batch, time]
        4.对output computer_score，mask过滤
        5.返回output, state(来自rnn), attn_weight
        '''
        if self.config.global_emb:
            batch_size = contexts.size(0)
            a, b = self.embedding.weight.size()
            if soft_score is None:  # 第一次使用的时候soft_score == None,使用原始embedding
                emb = self.embedding(input)
            else:
                # 按soft_score对embedding加权，soft来自上一时步的decoder rnn输出过attention再计算score
                emb1 = torch.bmm(soft_score.unsqueeze(1), self.embedding.weight.expand((batch_size, a, b)))
                emb2 = self.embedding(input)  # 这里input是beam_search找到的几个最优值，因此大致是符合论文说的概率最大的
                                              # 所以说是对beam_search的改进
                # gamma控制使用（综合前i-1步的最可能选项）或者（当前输出概率的加权平均）的比例
                gamma = F.sigmoid(self.gated1(emb1.squeeze())+self.gated2(emb2.squeeze()))
                emb = gamma * emb1.squeeze() + (1 - gamma) * emb2.squeeze()
        else:
            emb = self.embedding(input)
        # output [batch, hidden * num_directions]
        # h, c [num_layer * num_dir, batch, hidden_sz]
        output, state = self.rnn(emb, state)
        # attn_weights: [batch, time]
        hidden, attn_weights = self.attention(output, contexts)
        output = self.compute_score(hidden)
        if self.config.mask:
            if mask is not None:
                output = output.scatter_(1, mask, -999999999)
        return output, state, attn_weights
