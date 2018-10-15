# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import data.dict as dict
import models

import os
import numpy as np


class seq2seq(nn.Module):

    def __init__(self, config, src_vocab_size, tgt_vocab_size, use_cuda, pretrain=None, score_fn=None):
        super(seq2seq, self).__init__()
        if pretrain is not None:
            # hint: 会自动冻结
            src_embedding = nn.Embedding.from_pretrained(pretrain['src_emb'])
            tgt_embedding = nn.Embedding.from_pretrained(pretrain['tgt_emb'])

            # def normal2(A):
            #     return A / np.sqrt(np.sum(A ** 2))

            # for i in range(len(pretrain['tgt_emb'])):
            #     pretrain['tgt_emb'][i] = normal2(pretrain['tgt_emb'][i])
            # mat = np.zeros(45*45).reshape(45, 45)
            # for i in range(45):
            #     for j in range(45):
            #         _ = normal2(pretrain['tgt_emb'][i].numpy().copy())
            #         __ = normal2(pretrain['tgt_emb'][j].numpy().copy())
            #         mat[i][j] = _.dot(__)
            # print(mat)
            # print()
        else:
            src_embedding = None
            tgt_embedding = None
        self.encoder = models.rnn_encoder(config, src_vocab_size, embedding=src_embedding, tgt_embedding=tgt_embedding)
        if config.shared_vocab == False:
            self.decoder = models.rnn_decoder(config, tgt_vocab_size, embedding=tgt_embedding, score_fn=score_fn)
        else:
            self.decoder = models.rnn_decoder(config, tgt_vocab_size, embedding=self.encoder.embedding,
                                              score_fn=score_fn)
        self.use_cuda = use_cuda
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.config = config
        if config.score == 'margin':
            # print("using margin loss")
            self.criterion = models.margin_criterion(tgt_vocab_size, use_cuda, config)
        elif config.score == 'hybrid':
            self.criterion = {}
            self.criterion['softmax'] = models.criterion(tgt_vocab_size, use_cuda, config)
            self.criterion['margin'] = models.margin_criterion(tgt_vocab_size, use_cuda, config)
        elif config.score == 'hubness':
            self.criterion = models.mse_criterion(tgt_vocab_size, use_cuda, config)
        elif config.score == 'softmax':
            self.criterion = models.criterion(tgt_vocab_size, use_cuda, config)
        elif config.score == 'disc':
            self.criterion = {}
            self.criterion['softmax'] = models.criterion(2, use_cuda, config)
            self.criterion['margin'] = models.margin_criterion(tgt_vocab_size, use_cuda, config)
        else:
            print('no such score function')
            os.abort()
        self.log_softmax = nn.LogSoftmax(dim=1)

    # used in tarin to evaluate himself
    def compute_loss(self, hidden_outputs, targets, memory_efficiency, from_known):
        if memory_efficiency:
            return models.memory_efficiency_cross_entropy_loss(hidden_outputs, self.decoder, targets, self.criterion,
                                                               self.config)
        else:
            return models.cross_entropy_loss(hidden_outputs, self.decoder, targets, self.criterion, self.config, self, from_known)


    def forward(self, src, src_len, tgt, tgt_len, from_known):
        '''
        :param src: [maxlen, batch]
        :param src_len: [1, batch]
        :param tgt: [maxlen, batch]
        :param tgt_len: [1, batch]
        :return: outputs(shape[time, batch, dec_hidden]), target(shape[maxlen, batch])
                Arg outputs： decoder的输出再过一个attention得到
                Arg target： 就是输入的target减去开头的GO得到
        朴素的encoder和decoder结构，attetnion集成在了decoder中，由于使用双向rnn一定要pack, 并且句子长度需要为降序，
        因此这里需要对长度排序
        encoder:
            contexts [maxlen, batch, hidden*num_dirc]
            state -> (h, c) -> [num_layer, batch, dir * hidden]
        decoder；
            outputs [time, batch, hidden*num_dir] 所有时步的attention output
            state (h, c) [num_layer * num_dir, batch, hidden_sz] 最后一个时步的RNN状态
        '''
        lengths, indices = torch.sort(src_len.squeeze(0), dim=0, descending=True)  # src_len -> [batch], then sort
        src = torch.index_select(src, dim=1, index=indices)  # synchronize src with src_len
        tgt = torch.index_select(tgt, dim=1, index=indices)
        from_known = torch.index_select(from_known, dim=1, index=indices)

        import time
        try:
            print(src)
        except:
            time.sleep(10)
        try:
            print(lengths)
        except:
            time.sleep(10)
            
        contexts, state = self.encoder(src, lengths.data.tolist())  # contexts [maxlen, batch, hidden*num_dirc]
        # state -> (h, c) -> [num_layer, batch, dir * hidden]
        # input, initial_state, context
        # decoder的input应加上GO（这里也许加了？）并去除最后的EOS（因为EOS是希望rnn最后一个时步预测出来的，而不是我们输入给rnn的）
        outputs, final_state = self.decoder(tgt[:-1], state, contexts.transpose(0, 1))
        # outputs[time, batch, dec_hidden], final_state [num_layer, batch]
        return outputs, tgt[1:], from_known

    def sample(self, src, src_len):
        '''
        :param src: [maxlen, batch]
        :param src_len: [batch]
        :return:
        多gpu下采用的eval策略
        1.类似train的过程，排序，过encoder
            contexts [maxlen, batch, hidden*num_dirc]
            state -> (h, c) -> [num_layer, batch, dir * hidden]
        2.decoder.sample 只输入一个BOS,返回
            sample_ids shape[time, batch]
            outputs [time, batch, vocab_trg]
            attn_weights shape[time(dec), batch, time(enc)]
        3.恢复原有顺序
        4.返回sample_id(batch, time), alignment(batch, time(dec))
        '''
        if self.use_cuda:
            src = src.cuda()
            src_len = src_len.cuda()

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, ind = torch.sort(indices)  # returns newly_sorted, the ind from origin  # 1432 --ind-> 2341 --ind-> 3214
        src = Variable(torch.index_select(src, dim=1, index=indices), volatile=True)  # synchronize input
        bos = Variable(torch.ones(src.size(1)).long().fill_(dict.BOS), volatile=True)  # bos Variable[batch]

        if self.use_cuda:
            bos = bos.cuda()

        # contexts [maxlen, batch, hidden*num_dirc]  state -> (h, c) -> [num_layer, batch, dir * hidden]
        contexts, state = self.encoder(src, lengths.tolist())
        # sample_ids: 输出序列的id_list, (outputs, attns)  output:过attention后计算的score，attns:所有时步的attention weight [time(dec), batch, time(enc)]
        # sample_ids shape[time, batch]
        # outputs [time, batch, vocab_trg]
        # attn_weights shape[time(dec), batch, time(enc)]
        sample_ids, final_outputs = self.decoder.sample([bos], state, contexts.transpose(0, 1))  # hint: 从这里可以看出bos是不含在
        _, attns_weight = final_outputs
        alignments = attns_weight.max(2)[1]  # 在encoder中最匹配的位置
        sample_ids = torch.index_select(sample_ids.data, dim=1, index=ind)  # 恢复到与trg相同的顺序（取消句长顺序） shape[time, batch]
        alignments = torch.index_select(alignments.data, dim=1, index=ind)
        # targets = tgt[1:]

        return sample_ids.t(), alignments.t()  # (batch, time), (batch, time(dec))

    def beam_sample(self, src, src_len, beam_size=1):
        '''
        :param src: [maxlen, batch]
        :param src_len: [batch]
        :param beam_size: from config
        :return:
        1.句长排序，输入encoder，返回contexts，encState
            contexts [maxlen, batch, hidden*num_dirc]
            state -> (h, c) -> [num_layer, batch, dir * hidden]
        2.
        '''
        # beam_size = self.config.beam_size
        batch_size = src.size(1)

        # (1) Run the encoder on the src. Done!!!!
        if self.use_cuda:
            src = src.cuda()
            src_len = src_len.cuda()

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, ind = torch.sort(indices)
        with torch.no_grad():
            src = torch.index_select(src, dim=1, index=indices)
        # contexts [maxlen, batch, hidden*num_dirc]
        # state -> (h, c) -> [num_layer, batch, dir * hidden]
        contexts, encState = self.encoder(src, lengths.tolist())

        #  (1b) Initialize for the decoder.
        def var(a):
            with torch.no_grad():
                return a  # 相当于设置为inference模式

        def rvar(a):
            return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # Repeat everything beam_size times.
        contexts = rvar(contexts.data).transpose(0, 1)  # [len, batch, hid*dirc] -> [batch * beam_sz, len, hid*dirc]
        decState = (rvar(encState[0].data), rvar(encState[1].data))  # [layer, batch*bean_sz, dir*hidden]
        # decState.repeat_beam_size_times(beam_size)
        beam = [models.Beam(beam_size, n_best=1,
                            cuda=self.use_cuda)
                for __ in range(batch_size)]  # batch_sz个Beam对象

        # (2) run the decoder to generate sentences, using beam search.

        mask = None
        soft_score = None
        for i in range(self.config.max_tgt_len):

            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            # the nextYs in each b is initially [BOS] + [EOS] * beam_sz - 1
            inp = var(torch.stack([b.getCurrentState() for b in beam])  # b.getCurrentState return a longTensor of shape [beam_sz]
                                                                        # then stack it to [batch, beam_sz], t() -> [beam_sz, batch] -> [beam*batch]
                                                                        # 符合decoder的输入
                      .t().contiguous().view(-1))

            # Run one step. # inp[80], soft_score -> initially None  decState->[layer, batch*bean_sz, dir*hidden]
            # contexts -> [batch * beam_sz, len, hid*dirc], mask initially None
            # HINT: output就是score

            output, decState, attn = self.decoder.sample_one(inp, soft_score, decState, contexts, mask)
            soft_score = F.softmax(output, dim=1)
            
            '''
            import time
            print(soft_score[:3])
            print(output.max(1)[0])
            time.sleep(20)
            '''
            

            if self.config.view_score:
                import time
                print(output[:10])
                time.sleep(20)

            if self.config.cheat:
                output[:, -5:] += self.config.cheat
            predicted = output.max(1)[1]
            # predicted = torch.multinomial(soft_score, 1).squeeze()
            
            # print(output.shape, predicted.shape)
            # print(predicted[:8])
            # time.sleep(20)

            '''
            the_index = (torch.zeros(predicted.shape).float().cuda() + 0.1).cuda().gt(output.max(1)[0])
            # the_index = output.max(1)[0].gt(torch.zeros(predicted.shape).cuda() + 0.1).cuda()
            # print(the_index, predicted)
            predicted = predicted.masked_fill_(the_index, 3)
            '''
            
            '''
            import time
            print(predicted)
            time.sleep(20)
            '''

            if self.config.mask:
                if mask is None:
                    mask = predicted.unsqueeze(1).long()
                else:
                    mask = torch.cat((mask, predicted.unsqueeze(1)), 1)
            # decOut: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            output = unbottle(self.log_softmax(output))
            attn = unbottle(attn)
            # beam x tgt_vocab

            # (c) Advance each beam.
            # update state
            for j, b in enumerate(beam):
                b.advance(output.data[:, j], attn.data[:, j])
                b.beam_update(decState, j)

        # (3) Package everything up.
        allHyps, allScores, allAttn = [], [], []

        for j in ind:
            b = beam[j]
            n_best = 1
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att.max(1)[1])  # 最关注的位置
            allHyps.append(hyps[0])
            allScores.append(scores[0])
            allAttn.append(attn[0])

        # print(allHyps)
        # print(allAttn)
        return allHyps, allAttn
