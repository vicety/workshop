import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data

import models
import data.dataloader as dataloader
import data.utils as utils
import data.dict as dict
from optims import Optim
import lr_scheduler as L

import os
import argparse
import time
import json
import collections
from collections import OrderedDict
import codecs

if __name__ == '__main__':
    new_labels = ['bua', 'tlk', 'zah', 'sko', 'buk', '<unk>']  # 不需要空格
    #config
    parser = argparse.ArgumentParser(description='train.py')

    parser.add_argument('-config', default='config.yaml', type=str,
                        help="config file")
    parser.add_argument('-gpus', default=[3], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")
    '''
    parser.add_argument('-restore', default='checkpoint.pt', type=str,
                        help="restore checkpoint")
    
    '''
    parser.add_argument('-restore', default='', type=str,
                        help="restore checkpoint")
    
    parser.add_argument('-seed', type=int, default=514,
                        help="Random seed")
    parser.add_argument('-model', default='seq2seq', type=str,
                        help="Model selection")
    parser.add_argument('-score', default='disc', type=str,  # 影响网络结构
                        help="score_fn")
    parser.add_argument('-pretrain', default=True, type=bool,
                        help="load pretrain embedding")
    parser.add_argument('-notrain', default=False, type=bool,
                        help="train or not")
    parser.add_argument('-limit', default=0, type=int,
                        help="data limit")
    parser.add_argument('-log', default='', type=str,
                        help="log directory")
    parser.add_argument('-unk', default=False, type=bool,
                        help="replace unk")
    parser.add_argument('-memory', default=False, type=bool,
                        help="memory efficiency")
    parser.add_argument('-label_dict_file', default='./data/data/target_label_dict.json', type=str,
                        help="label_dict")
    parser.add_argument('-label_dict_test_file', default='./data/data/test_label_dict.json', type=str,
                        help="label_dict")

    opt = parser.parse_args()
    config = utils.read_config(opt.config)
    torch.manual_seed(opt.seed)

    if not os.path.exists(config.log):
        os.mkdir(config.log)
    if opt.log == '':
        log_path = config.log + str(utils.format_time(time.localtime())).replace(':', '_') + '/'
    else:
        log_path = config.log + opt.log + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    logging = utils.logging(log_path+'log.txt')

    # checkpoint
    if opt.restore:
        print('loading checkpoint...\n')

        checkpoints = torch.load(opt.restore, map_location='cpu')

    # cuda
    use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
    #use_cuda = True
    if use_cuda:
        torch.cuda.set_device(opt.gpus[0])
        torch.cuda.manual_seed(opt.seed)
    print(use_cuda)

    # data
    print('loading data...\n')
    start_time = time.time()
    datas = torch.load(config.data)
    print('loading time cost: %.3f' % (time.time()-start_time))

    trainset, validset, testset = datas['train'], datas['valid'], datas['test']
    src_vocab, tgt_vocab = datas['dicts']['src'], datas['dicts']['tgt']
    print(tgt_vocab.idxToLabel)
    config.src_vocab = src_vocab.size()
    config.tgt_vocab = tgt_vocab.size()

    trainloader = dataloader.get_loader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    # print(trainloader.dataset.from_known)

    f = open('./data/data/train_label.txt', 'r')
    lines = f.readlines()

    train_dataset = trainloader.dataset
    new_id = list(map(lambda x: int(tgt_vocab.labelToIdx[x + ' ']), new_labels))
    # print(list(new_id))
    for i in range(len(train_dataset.from_known)):
        flag = 0
        for label in train_dataset.tgt[i]:
            # print(label.item(), new_id)
            if label.item() in new_id:
                flag = 1
        if flag:
            assert train_dataset.from_known[i][0] is not True
        else:
            #
            assert train_dataset.from_known[i][0] is True

    validloader = dataloader.get_loader(validset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    testloader = dataloader.get_loader(testset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    if opt.pretrain:
        pretrain_embed = torch.load(config.emb_file)
        logging('using pretrained embedding\n')
    else:
        pretrain_embed = None

    # model
    print('building model...\n')
    model = getattr(models, opt.model)(config, src_vocab.size(), tgt_vocab.size(), use_cuda,
                           pretrain=pretrain_embed, score_fn=opt.score)

    if opt.restore:
        state_dict = OrderedDict()
        for k, v in model.state_dict().items():
            # print(k)
            if(type(checkpoints['model'].get(k)) == type(torch.Tensor(1))):
                state_dict[k] = checkpoints['model'][k]
            else:
                state_dict[k] = v

        model.load_state_dict(state_dict)

        # for k, v in model.state_dict().items():
        #     if k == 'decoder.rnn.layers.1.weight_ih':
        #         v.requires_grad = True
        #     else:
        #         v.requires_grad = False

        # if config.global_emb:  # ALL SAME
        # if config['freeze']:
        #     keep_alive = [26, 27]
        #
        #     for i, p in enumerate(model.parameters()):
        #         # HINT: 如果全部冻结,那么计算loss那行不会返回任何东西
        #         # HINT: 如同下一行的测试，这里保留了上一个网络需要冻结的东西，其他都require_grad=True
        #         # print(p.requires_grad)
        #         if not i in keep_alive:
        #             p.requires_grad = False
        #         else:
        #             print('here')
        #             pass

    # for i, p in enumerate(model.parameters()):
    #     logging('layer {}, requires_grad={}'.format(p, p.requires_grad))

    # for i, (k, v) in enumerate(model.state_dict().items()):
    #     print(k, v.shape)
    #     print(i)
    #
    # print('----')
    #
    # for i, p in enumerate(model.parameters()):
    #     print(i)
    #     print(p.shape)

    if use_cuda:
        model.cuda()
    if len(opt.gpus) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)

    # optimizer
    if opt.restore:
        optim = checkpoints['optim']
    else:
        optim = Optim(config.optim, config.learning_rate, config.max_grad_norm,
                      lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)
    optim.set_parameters(filter(lambda p: p.requires_grad,model.parameters()))
    if config.schedule:
        scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)

    # total number of parameters
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]



    logging_csv = utils.logging_csv(log_path+'record.csv')
    for k, v in config.items():
        logging("%s:\t%s\n" % (str(k), str(v)))
    logging("\n")
    logging(repr(model)+"\n\n")

    logging('total number of parameters: %d\n\n' % param_count)
    logging('score function is %s\n\n' % opt.score)

    if opt.restore:
        updates = checkpoints['updates']
    else:
        updates = 0

    total_loss, start_time = 0, time.time()
    report_total, report_correct = 0, 0
    report_vocab, report_tot_vocab = 0, 0
    scores = [[] for metric in config.metric]
    scores = collections.OrderedDict(zip(config.metric, scores))
    test_scores = [[] for metric in config.metric]
    test_scores = collections.OrderedDict(zip(config.metric, test_scores))

    with open(opt.label_dict_file, 'r') as f:
        label_dict = json.load(f)

    with open(opt.label_dict_test_file, 'r') as f:
        label_dict_test = json.load(f)

# train
def train(epoch):
    global e, updates, total_loss, start_time, report_total

    e = epoch
    model.train()  # switch to tarin model

    if config.schedule:
        scheduler.step()
        print("Decaying learning rate to %g" % scheduler.get_lr()[0])

    if opt.model == 'gated': 
        model.current_epoch = epoch

    # raw_tgt无用
    for raw_src, src, src_len, raw_tgt, tgt, tgt_len, from_known in trainloader:
        # from_known [batch, 1]
        src = Variable(src)
        tgt = Variable(tgt)
        src_len = Variable(src_len).unsqueeze(0)  # add one dimension at first, get [1, sentence_num]
        tgt_len = Variable(tgt_len).unsqueeze(0)

        # if 增加随机前缀，那么同时还要修改tgt_len

        if use_cuda:
            src = src.cuda()
            tgt = tgt.cuda()
            src_len = src_len.cuda()
            tgt_len = tgt_len.cuda()
            from_known = from_known.cuda()

        model.zero_grad()
        # outputs [batch, dec_hidden], targets 句子去掉GO [maxlen, batch]
        # print(model.decoder.rnn.layers._modules['0']._parameters['weight_ih'].data)
        outputs, targets, from_known = model(src, src_len, tgt, tgt_len, from_known)  # execute forward computation here
        '''
        import time
        print(targets[:3])
        time.sleep(10)
        '''

        # 对于每一列assert其from_known的正确性
        maxlen, batch_size = targets.shape
        for i in range(batch_size):
            batch_labels = targets[:, i]
            if from_known[i]:
                for label_id in list(map(lambda x: int(tgt_vocab.labelToIdx[x + ' ']), new_labels)):
                    assert label_id not in batch_labels
            else:
                flag = False
                for label_id in list(map(lambda x: int(tgt_vocab.labelToIdx[x + ' ']), new_labels)):
                    if label_id in batch_labels:
                        flag = True
                assert flag is True

        loss, num_total, _, _, _ = model.compute_loss(outputs, targets, opt.memory, from_known)

        if config.batch_size == 1 :
            label_from_test = new_labels + ['<unk>']
            if from_known[0] == 0:
                assert loss == 0
                flag = False
                for label in label_from_test:
                    if label in raw_tgt[0]:
                        flag = True
                assert flag is True
            else:
                flag = False
                for label in label_from_test:
                    if label in raw_tgt[0]:
                        flag = True
                assert flag is False



        # TODO: make sure what the num_total and the follow _ _ _ is about.
        total_loss += loss
        report_total += num_total
        optim.step()  # clip the gradient then apply it
        updates += 1
        if updates % 100 == 0:
            print(updates)

        if updates % config.eval_interval == 0:
            logging("time: %6.3f, epoch: %3d, updates: %8d, train loss: %6.3f\n"
                    % (time.time()-start_time, epoch, updates, total_loss / report_total))
            print('evaluating after %d updates...\r' % updates)
            score = eval(epoch)

            for metric in config.metric:
                scores[metric].append(score[metric])  # seems to be double itself, why ?
                if metric == 'micro_f1' and score[metric] >= max(scores[metric]):  
                    save_model(log_path+'best_'+metric+'_checkpoint.pt')
                if metric == 'hamming_loss' and score[metric] <= min(scores[metric]):
                    save_model(log_path+'best_'+metric+'_checkpoint.pt')

            test_score = eval_test(epoch)
            for metric in config.metric:
                test_scores[metric].append(test_score[metric])  # seems to be double itself, why ?
                if metric == 'micro_f1' and test_score[metric] >= max(test_scores[metric]):
                    save_model(log_path+'test_best_'+metric+'_checkpoint.pt')
                if metric == 'hamming_loss' and test_score[metric] <= min(test_scores[metric]):
                    save_model(log_path+'test_best_'+metric+'_checkpoint.pt')

            model.train()
            total_loss = 0
            # start_time = 0
            report_total = 0

        if updates % config.save_interval == 0:  
            save_model(log_path+'checkpoint.pt')


def eval(epoch):
    model.eval()
    reference, candidate, source, alignments = [], [], [], []
    for raw_src, src, src_len, raw_tgt, tgt, tgt_len, from_known in validloader:
        if len(opt.gpus) > 1:
            samples, alignment = model.module.sample(src, src_len)
        else:
            samples, alignment = model.beam_sample(src, src_len, beam_size=config.beam_size)

        candidate += [tgt_vocab.convertToLabels(s, dict.EOS) for s in samples]
        source += raw_src
        reference += raw_tgt
        alignments += [align for align in alignment]

    if opt.unk:
        cands = []
        for s, c, align in zip(source, candidate, alignments):
            cand = []
            for word, idx in zip(c, align):
                if word == dict.UNK_WORD and idx < len(s):
                    try:
                        cand.append(s[idx])
                    except:
                        cand.append(word)
                        print("%d %d\n" % (len(s), idx))
                else:
                    cand.append(word)
            cands.append(cand)
        candidate = cands

    score = {}
    result = utils.eval_metrics(reference, candidate, label_dict, log_path)
    logging_csv([e, updates, result['hamming_loss'], \
                result['micro_f1'], result['micro_precision'], result['micro_recall']])
    print('hamming_loss: %.8f | micro_f1: %.4f'
          % (result['hamming_loss'], result['micro_f1']))
    score['hamming_loss'] = result['hamming_loss']
    score['micro_f1'] = result['micro_f1']
    return score


def eval_test(epoch):
    model.eval()
    reference, candidate, source, alignments = [], [], [], []
    for raw_src, src, src_len, raw_tgt, tgt, tgt_len, from_known in testloader:
        if len(opt.gpus) > 1:
            samples, alignment = model.module.sample(src, src_len)
        else:
            samples, alignment = model.beam_sample(src, src_len, beam_size=config.beam_size)

        test_vocab = tgt_vocab
        for label in new_labels:
            test_vocab.add(label + ' ')

        candidate += [test_vocab.convertToLabels(s, dict.EOS) for s in samples]
        source += raw_src
        reference += raw_tgt
        alignments += [align for align in alignment]

    if opt.unk:
        cands = []
        for s, c, align in zip(source, candidate, alignments):
            cand = []
            for word, idx in zip(c, align):
                if word == dict.UNK_WORD and idx < len(s):
                    try:
                        cand.append(s[idx])
                    except:
                        cand.append(word)
                        print("%d %d\n" % (len(s), idx))
                else:
                    cand.append(word)
            cands.append(cand)
        candidate = cands

    score = {}
    result = utils.eval_metrics(reference, candidate, label_dict_test, log_path)
    logging_csv(["testing", e, updates, result['hamming_loss'], \
                result['micro_f1'], result['micro_precision'], result['micro_recall']])
    print('In testing: hamming_loss: %.8f | micro_f1: %.4f'
          % (result['hamming_loss'], result['micro_f1']))
    score['hamming_loss'] = result['hamming_loss']
    score['micro_f1'] = result['micro_f1']
    return score


def save_model(path):
    global updates
    model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,  # config
        'optim': optim,  # optimizer
        'updates': updates}  # update: how many steps now
    torch.save(checkpoints, path)


def main():
    for i in range(1, config.epoch+1):
        if not opt.notrain:
            train(i)
        else:
            eval(i)
    for metric in config.metric:
        logging("Best %s score: %.2f\n" % (metric, max(scores[metric])))


if __name__ == '__main__':
    main()
