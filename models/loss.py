# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import models
import data.dict as dict
from torch.autograd import Variable


def criterion(tgt_vocab_size, use_cuda, config):
    weight = torch.ones(tgt_vocab_size)
    weight[dict.PAD] = 0
    crit = nn.CrossEntropyLoss(weight, size_average=False)
    if use_cuda:
        crit.cuda()
    return crit


# def mse_criterion(tgt_vocab_size, use_cuda, config):
#     def mse(scores, target):
#         value, arg = torch.max(scores)
#         target = target.unsqueeze(1)
#

# def margin_criterion_with_mask(tgt_vocab_size, use_cuda, config):
#     def hinge_margin_loss(scores, target, from_known):
#         '''
#         :param scores: [time*batch, vocab],
#         :param target: [maxlen*batch]
#         :return: loss:float
#         '''
#         if use_cuda:
#             mask = torch.ones(scores.shape).cuda()
#             mask[:, dict.PAD] = 0
#             for i in range(scores.shape[0]):
#                 mask[i][target[i]] = 0
#             error_selection_index = torch.multinomial(mask, config['N'])  # [time*batch, N]
#             error_score = torch.gather(scores, 1, error_selection_index).cuda()
#             error_score = error_score.sum(dim=1)
#             correct_score = torch.gather(scores, 1, target.unsqueeze(1)).squeeze().cuda()
#             loss = config['margin'] + config['err_mul'] * error_score - correct_score
#             loss_index = loss.gt(torch.zeros(scores.shape[0]).cuda()).cuda()
#             # TODO: mask here
#             loss = torch.masked_select(loss, loss_index).sum(dim=0).cuda()
#         else:
#             print('buy a gpu...')
#
#         if config["view_score"]:
#             import time
#             sample = 1
#             # 缺陷在于不能看到target
#             print(scores.shape)
#             print("scores[:10]: {}\n".format(scores[:10]))
#             print("target: {}".format(target.view(-1, config['batch_size'])))
#             print("err[:10]: {}\n cor[:10]: {}\n".format(error_score[:10], correct_score[:10]))
#             time.sleep(10)
#         return loss
#
#     return hinge_margin_loss

def margin_criterion(tgt_vocab_size, use_cuda, config):
    def hinge_margin_loss(scores, target, from_known=None):
        '''
        :param scores: [time*batch, vocab],
        :param target: [maxlen*batch]
        :return: loss:float
        '''
        if use_cuda:
            mask = torch.ones(scores.shape).cuda()
            mask[:, dict.PAD] = 0
            for i in range(scores.shape[0]):
                mask[i][target[i]] = 0
            error_selection_index = torch.multinomial(mask, config['N'])  # [time*batch, N]
            error_score = torch.gather(scores, 1, error_selection_index).cuda()
            error_score = error_score.sum(dim=1)
            correct_score = torch.gather(scores, 1, target.unsqueeze(1)).squeeze().cuda()
            loss = config['margin'] + config['err_mul'] * error_score - correct_score
            loss_index = loss.gt(torch.zeros(scores.shape[0]).cuda()).cuda()
            if from_known is not None:
                sentence_len = int(error_score.shape[0] / config.batch_size)
                from_known = from_known.t().expand(sentence_len, config.batch_size).cuda().view(-1)
                loss *= from_known
            loss = torch.masked_select(loss, loss_index).sum(dim=0).cuda()
        else:
            mask = torch.ones(scores.shape)
            mask[:, dict.PAD] = 0
            for i in range(scores.shape[0]):
                mask[i][target[i]] = 0
                error_selection_index = torch.multinomial(mask, config['N'])  # [time*batch, N]
                error_score = torch.gather(scores, 1, error_selection_index)
                error_score = error_score.sum(dim=1)
                correct_score = torch.gather(scores, 1, target.unsqueeze(1)).squeeze()
                loss = config['margin'] + error_score - correct_score
                loss_index = loss.gt(torch.zeros(scores.shape[0]))
                loss = torch.masked_select(loss, loss_index).sum(dim=0)

        if config["view_score"]:
            import time
            sample = 1
            # 缺陷在于不能看到target
            print(scores.shape)
            print("scores[:10]: {}\n".format(scores[:10]))
            print("target: {}".format(target.view(-1, config['batch_size'])))
            print("err[:10]: {}\n cor[:10]: {}\n".format(error_score[:10], correct_score[:10]))
            time.sleep(10)
        return loss

    return hinge_margin_loss

def memory_efficiency_cross_entropy_loss(hidden_outputs, decoder, targets, criterion, config):
    outputs = Variable(hidden_outputs.data, requires_grad=True, volatile=False)
    num_total, num_correct, loss = 0, 0, 0

    outputs_split = torch.split(outputs, config.max_generator_batches)
    targets_split = torch.split(targets, config.max_generator_batches)
    for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
        out_t = out_t.view(-1, out_t.size(2))
        scores_t = decoder.compute_score(out_t)
        loss_t = criterion(scores_t, targ_t.view(-1))
        pred_t = scores_t.max(1)[1]
        num_correct_t = pred_t.data.eq(targ_t.data).masked_select(targ_t.ne(dict.PAD).data).sum()
        num_total_t = targ_t.ne(dict.PAD).data.sum()
        num_correct += num_correct_t
        num_total += num_total_t
        loss += loss_t.data[0]
        loss_t.div(num_total_t).backward()

    grad_output = outputs.grad.data
    hidden_outputs.backward(grad_output)

    return loss, num_total, num_correct, config.tgt_vocab, config.tgt_vocab


# outputs [time, batch, dec_hidden], targets 句子去掉GO [maxlen, batch]
def cross_entropy_loss(hidden_outputs, decoder, targets, criterion, config, model, from_known, sim_score=0):
    '''
    :param hidden_outputs: [time, batch, dec_hidden]
    :param decoder: self.decoder
    :param targets: [maxlen, batch] targets 句子去掉GO
    :param criterion: here nn.CrossEntropyLoss loss衡量方式
    :param config:
    :param sim_score:
    :param from_known: [batch, 1] 也许吧
    :return:
    decoder的输出经过decoder.computer_score后按某种方式（这里是选score最高的）选择预测结果计算loss，并apply gradient
    '''
    outputs = hidden_outputs.view(-1, hidden_outputs.size(2))  # outputs [time * batch, dec_hidden]
    scores = decoder.compute_score(outputs)
    if config.score == 'hinge_margin_loss' or config.score == 'softmax':
        # TODO: 去看一下RNN怎么做backward的
        loss = criterion(scores, targets.view(-1)) + sim_score  # ([time*batch, vocab], [maxlen*batch])
        # print(loss)
        pred = scores.max(1)[1]  # max(dim), 返回（最大值，对应index）  # TODO: 这里也可以尝试按概率来返回pred（或者更复杂一些）？
    elif config.score == 'hybrid':
        softmax_loss = criterion['softmax'](scores['softmax'], targets.view(-1)) + sim_score  # ([time*batch, vocab], [maxlen * batch])
        margin_loss = criterion['margin'](scores['margin'], targets.view(-1)) + sim_score
        print('softmax_loss: {} || margin_loss: {}'.format(softmax_loss, margin_loss))
        loss = softmax_loss * config['softmax_linear_lr_mul'] + margin_loss
        pred = scores['margin'].max(1)[1]
    elif config.score == 'hubness':
        mse_loss = criterion(scores, targets.view(-1)) + sim_score
        l2_reg = config['l2_reg'] * ((model.decoder.emb_to_mid1.weight.data ** 2).sum()
                                     + (model.decoder.mid1_to_mid2.weight.data ** 2).sum())
        print("mse_loss: {} || l2_reg: {}".format(mse_loss, l2_reg))
        loss = mse_loss + l2_reg
        pred = scores.max(1)[1]
    elif config.score == 'disc':
        # hint: 也就是说每个时步的decoder output都要拿来算一次是否来自train
        sentence_len = int(scores['margin'].shape[0] / config.batch_size)
        softmax_loss = (criterion['softmax'](scores['softmax'],
                                        from_known.t().expand(sentence_len, config.batch_size).cuda().long().view(-1))\
                                        + sim_score) * config['softmax_linear_lr_mul']   # ([time*batch, vocab], [maxlen * batch])
        margin_loss = criterion['margin'](scores['margin'], targets.view(-1), from_known) + sim_score
        print('softmax_loss: {} || margin_loss: {}'.format(softmax_loss, margin_loss))
        loss = softmax_loss + margin_loss
        pred = scores['margin'].max(1)[1]
    num_correct = pred.data.eq(targets.view(-1).data)
        # padding的正确率不计
    num_correct = num_correct.masked_select(targets.view(-1).ne(dict.PAD).data)
    num_correct = num_correct.sum()
    num_total = targets.ne(dict.PAD).data.sum().float()  # ne 是否相等，不等为1，算总共单词数
    loss.div(num_total).backward()
    loss = loss.item()  

    return loss, num_total, num_correct, config.tgt_vocab, config.tgt_vocab
