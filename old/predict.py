import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import models
import data.dataloader as dataloader
import data.utils as utils
import data.dict as dict
from optims import Optim
from stat_predict import print_stat

from optims import Optim
import lr_scheduler as L

import os
import argparse
import time
import math
import json
import collections
from collections import OrderedDict
import codecs
import numpy as np

if __name__ == '__main__':
    new_labels = ['bua ', 'tlk ', 'zah ', 'sko ', 'buk ']  # 记得加空格
    # config
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('-config', default='config.test.yaml', type=str,
                        help="config file")
    parser.add_argument('-gpus', default=[3], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-restore', default='./checkpoint.pt', type=str,  # best_micro_f1_
                        help="restore checkpoint")
    parser.add_argument('-seed', type=int, default=514,
                        help="Random seed")
    parser.add_argument('-model', default='seq2seq', type=str,
                        help="Model selection")
    parser.add_argument('-score', default='hybrid', type=str,  # 影响compute_score
                        help="score_fn")
    parser.add_argument('-pretrain', default=True, action='store_true',
                        help="load pretrain embedding")
    parser.add_argument('-limit', type=int, default=0,
                        help="data limit")
    parser.add_argument('-log', default='predict', type=str,
                        help="log directory")
    parser.add_argument('-unk', default=False, action='store_true',
                        help="replace unk")
    parser.add_argument('-memory', action='store_true',
                        help="memory efficiency")
    parser.add_argument('-beam_size', type=int, default=1,
                        help="beam search size")
    parser.add_argument('-label_dict_file', default='data/data/test_label_dict.json', type=str,
                        help="label_dict")

    opt = parser.parse_args()
    config = utils.read_config(opt.config)
    torch.manual_seed(opt.seed)

    # checkpoint
    if opt.restore:
        print('loading checkpoint...\n')
        checkpoints = torch.load(opt.restore)

    # cuda
    use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
    # use_cuda = True
    if use_cuda:
        torch.cuda.set_device(opt.gpus[0])
        torch.cuda.manual_seed(opt.seed)

    # data
    print('loading data...\n')
    start_time = time.time()
    datas = torch.load(config.data)
    print('loading time cost: %.3f' % (time.time() - start_time))

    testset = datas['test']
    src_vocab, tgt_vocab = datas['dicts']['src'], datas['dicts']['tgt']

    for label in new_labels:
        tgt_vocab.add(label)

    config.src_vocab = src_vocab.size()
    config.tgt_vocab = tgt_vocab.size()
    testloader = dataloader.get_loader(testset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    if opt.pretrain:
        pretrain_embed = torch.load(config.emb_file)
        print(pretrain_embed['src_emb'].shape, pretrain_embed['tgt_emb'].shape)
    else:
        pretrain_embed = None

    # model
    print('building model...\n')
    # Notice: 这里的tgt_vocab size应该加上额外单词数
    model = getattr(models, opt.model)(config, src_vocab.size(), tgt_vocab.size(), use_cuda,
                                       pretrain=pretrain_embed, score_fn=opt.score)

    print(repr(model) + "\n\n")

    # 由于训练时的softmax不预测新类，而这里的softmax预测新类，因此不能用checkpoint的
    use_test_embedding = ['decoder.embedding.weight', 'encoder.tgt_embedding.weight',
                          'decoder.toCatLinear.weight', 'decoder.toCatLinear.bias',
                          'criterion.weight']  # 使用载入的embedding和默认的softmax前linear
    if opt.restore:
        state_dict = OrderedDict()
        for k, v in model.state_dict().items():
            if k in use_test_embedding:
                print("use default: {}".format(k))
                state_dict[k] = v
                continue
            if (type(checkpoints['model'].get(k)) == type(torch.Tensor(1))):
                state_dict[k] = checkpoints['model'][k]
            else:
                print("use_default: {}".format(k))
                state_dict[k] = v

        model.load_state_dict(state_dict)

    if use_cuda:
        model.cuda()
    if len(opt.gpus) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)

    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]

    if not os.path.exists(config.log):
        os.mkdir(config.log)
    if opt.log == '':
        log_path = config.log + utils.format_time(time.localtime()) + '/'
    else:
        log_path = config.log + opt.log + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logging = utils.logging(log_path + 'model_config.txt')
    logging_csv = utils.logging_csv(log_path + 'record.csv')
    for k, v in config.items():
        logging("%s:\t%s\n" % (str(k), str(v)))
    logging("\n")
    logging(repr(model) + "\n\n")

    logging('total number of parameters: %d\n\n' % param_count)

    with open(opt.label_dict_file, 'r') as f:
        label_dict = json.load(f)


def eval(epoch):
    model.eval()
    reference, candidate, source, alignments = [], [], [], []
    for raw_src, src, src_len, raw_tgt, tgt, tgt_len in testloader:
        if len(opt.gpus) > 1:
            samples, alignment = model.sample(src, src_len)
        else:
            # HINT: 对于beam来说 sample和align的长度相等
            samples, alignment = model.beam_sample(src, src_len, beam_size=config.beam_size)
            # print(samples[:2])

        # print([tgt_vocab.convertToLabels(s, dict.EOS) for s in samples][:2])
        # print(tgt_vocab.idxToLabel)
        # print('here')
        candidate += [tgt_vocab.convertToLabels(s, dict.EOS) for s in samples]
        # print(tgt_vocab.convertToLabels([torch.Tensor(35).long().cuda(), torch.Tensor(3).long().cuda()], dict.EOS))
        # print(candidate[-2:])
        source += raw_src
        reference += raw_tgt
        alignments += [align for align in alignment]
        print('.')

        # for i in range(20, 30):
        #     print(candidate[i])
        #     for align in alignment[i]:
        #         print(raw_src[i][align])

    if opt.unk:
        cands = []
        for s, c, align in zip(source, candidate, alignments):
            cand = []
            for word, idx in zip(c, align):
                if word == dict.UNK_WORD and idx < len(s):
                    try:
                        cand.append(s[idx])
                        # print("replace with {}".format(s[idx]))
                    except:
                        cand.append(word)
                        print("%d %d\n" % (len(s), idx))
                else:
                    cand.append(word)
            cands.append(cand)
        candidate = cands

    score = {}

    # alignment analysis
    with open("alignment_analysis.txt", 'w', encoding='utf-8') as f:
        # convert alignments to human readable words
        # oor_cnt = 0
        # global oor_cnt
        def map_align_to_word_(i):
            def map_align_to_word(align):
                if align < len(source[i]):
                    return source[i][align]
                else:
                    # oor_cnt += 1
                    return 'None'

            return map_align_to_word

        lines = []
        for i in range(len(alignments)):
            line = ""
            align_word = list(map(map_align_to_word_(i), alignments[i]))
            tgt_word = list(zip(candidate[i], align_word))
            for item in tgt_word:
                line += '({}, {}) '.format(item[0], item[1])
            line += '\n'
            lines += line
        f.writelines(lines)

    result = utils.eval_metrics(reference, candidate, label_dict, log_path)
    logging_csv([result['hamming_loss'], result['micro_f1'], result['micro_precision'], result['micro_recall']])
    print('hamming_loss: %.8f | micro_f1: %.4f'
          % (result['hamming_loss'], result['micro_f1']))


if __name__ == '__main__':
    eval(0)
    print_stat(len(new_labels))
