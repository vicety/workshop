# -*- coding: utf-8 -*-
import argparse
import torch
import data.dict as dict
from data.dataloader import dataset
from nltk import sent_tokenize
import json
from gensim.models import FastText
from gensim.models import KeyedVectors
import numpy as np
import data.utils as utils

new_label = ['bua', 'tlk', 'zah', 'sko', 'buk']   # 不需要空格
parser = argparse.ArgumentParser(description='preprocess.py')

##
## **Preprocess Options**
##

parser.add_argument('-config', default='config.yaml', type=str,
                    help="config file")

parser.add_argument('-train_src',
                    default='./data/data/train_doc.txt',
                    help="Path to the training source data")
parser.add_argument('-train_tgt',
                    default='./data/data/train_label.txt',
                    help="Path to the training target data")
parser.add_argument('-valid_src',
                    default='./data/data/dev_doc.txt',
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt',
                    default='./data/data/dev_label.txt',
                    help="Path to the validation target data")
parser.add_argument('-test_src',
                    default='./data/data/test_doc.txt',
                    help="Path to the validation source data")
parser.add_argument('-test_tgt',
                    default='./data/data/test_label.txt',
                    help="Path to the validation target data")

parser.add_argument('-save_data',
                    default='./data/data/save_data',
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=150,
                    help="Size of the target vocabulary")
parser.add_argument('-src_vocab', default=None,
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab', default=None,
                    help="Path to an existing target vocabulary")


parser.add_argument('-src_length', type=int, default=750,
                    help="Maximum source sequence length")
parser.add_argument('-tgt_length', type=int, default=25,
                    help="Maximum target sequence length")
parser.add_argument('-shuffle',    type=int, default=0,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', default = True, action='store_true', help='lowercase data')
parser.add_argument('-char', default = False, action='store_true', help='replace unk with char')
parser.add_argument('-share', default = False, action='store_true', help='share the vocabulary between source and target')

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()
config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)

'''
makeVocabulary: 
'''

def makeVocabulary(filename, size, char=False, ):
    # hint: 可见所有的Dict初始都含有special_words
    vocab = dict.Dict([dict.PAD_WORD, dict.UNK_WORD,
                        dict.BOS_WORD, dict.EOS_WORD], lower=opt.lower)


    if char:
        vocab.addSpecial(dict.SPA_WORD)

    lengths = []

    if type(filename) == list:
        for _filename in filename:
            with open(_filename) as f:
                for sent in f.readlines():
                    for word in sent.strip().split():
                        lengths.append(len(word))
                        if char:
                            for ch in word:
                                vocab.add(ch)
                        else:
                            vocab.add(word + " ")
    else:
        with open(filename, encoding='utf-8') as f:
            for sent in f.readlines():
                for word in sent.strip().split():
                    lengths.append(len(word))
                    if char:
                        for ch in word:
                            vocab.add(ch)
                    else:
                        vocab.add(word+" ")

    print('max: %d, min: %d, avg: %.2f' % (max(lengths), min(lengths), sum(lengths)/len(lengths)))

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFile, vocabFile, vocabSize, char=False):
    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = dict.Dict()
        vocab.loadFile(vocabFile)
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFile, vocabSize, char=char)
        vocab = genWordVocab

    print()
    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(srcFile, tgtFile, srcDicts, tgtDicts, sort=False, char=False):
    src, tgt, from_train = [], [], []
    raw_src, raw_tgt = [], []
    sizes = []
    count, ignored = 0, 0

    print('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile, encoding='utf-8')
    tgtF = open(tgtFile, encoding='utf-8')

    not_in_train_cnt = 0

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            print('WARNING: source and target do not have the same number of sentences')
            break

        sline = sline.strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            ignored += 1
            continue

        if opt.lower:
            sline = sline.lower()
            tline = tline.lower()

        srcWords = sline.split()
        tgtWords = tline.split()



        # 句长在限制范围内
        if opt.src_length == 0 or (len(srcWords) <= opt.src_length and len(tgtWords) <= opt.tgt_length):

            if char:
                srcWords = [word + " " for word in srcWords]
                tgtWords = list(" ".join(tgtWords))
            else:
                srcWords = [word+" " for word in srcWords]
                tgtWords = [word+" " for word in tgtWords]

            # 以id组成的句子
            src += [srcDicts.convertToIdx(srcWords,
                                          dict.UNK_WORD)]
            # target句加入了GO和EOS
            tgt += [tgtDicts.convertToIdx(tgtWords,
                                          dict.UNK_WORD,
                                          dict.BOS_WORD,
                                          dict.EOS_WORD)]
            # 原句（当然前面的lowercase还是在的）
            raw_src += [srcWords]
            raw_tgt += [tgtWords]
            sizes += [len(srcWords)]

            in_train = True
            for word in new_label + ['<unk>']:
                if word in map(lambda x: x.strip(), tgtWords):
                    in_train = False
                    not_in_train_cnt += 1
            from_train += [[in_train]]

        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()

    print(not_in_train_cnt)

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]
        raw_src = [raw_src[idx] for idx in perm]
        raw_tgt = [raw_tgt[idx] for idx in perm]
        from_train = [from_train[idx] for idx in perm]

    if sort:
        print('... sorting sentences by size')
        _, perm = torch.sort(torch.Tensor(sizes))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        raw_src = [raw_src[idx] for idx in perm]
        raw_tgt = [raw_tgt[idx] for idx in perm]
        from_train = [from_train[idx] for idx in perm]

    print('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
          (len(src), ignored, opt.src_length))

    
    new_id = [57, 58, 59, 60, 56, 1]
    for i in range(len(from_train)):
        flag = 0
        for label in tgt[i]:
            if label in new_id:
                flag = 1
        if flag:
            assert from_train[i][0] is not True
        else:
            assert from_train[i][0] is True

    dtst = dataset(src, tgt, raw_src, raw_tgt, from_train)

    print('len tgt {}, known {}'.format(len(dtst.tgt), len(dtst.from_known)))

    for i in range(len(dtst.from_known)):
        flag = 0
        for label in dtst.tgt[i]:
            if label in new_id:
                flag = 1
        if flag:
            print(i, dtst.tgt[i], dtst.from_known[i][0])
            assert dtst.from_known[i][0] is not True
        else:
            print(i, dtst.tgt[i], dtst.from_known[i][0])  # hint: 发现train只有10534
            assert dtst.from_known[i][0] is True
    # import time
    # time.sleep(10)

    return dtst


def toTranscz():
    di = {'<blank>': 'prázdné',
          '<unk>': 'neznámý',
          '<s>': 'začíná',
          '</s>': 'konec',
          'mak': 'makroekonomika',
          'slz': 'služby',
          'tur': 'cestovní ruch',
          'prg': 'Pragensie',
          'fin': 'finanční služby',
          'hok': 'Hokej - Zprávy',
          'eur': 'evropská unie - zpravodajství',
          'dpr': 'Doprava',
          'met': 'počasí',
          'pol': 'Politika',
          'zak': 'kriminalita a zákon',
          'for': 'parlamenty a vlády',
          'sta': 'Stavebnictví a reality',
          'efm': 'Firmy',
          'spo': 'sports',
          'sko': 'Školství',
          'med': 'média a reklama',
          'mag': 'časopis výběr',
          'spl': 'životní styl',
          'odb': 'práce a odbory',
          'pit': 'Telekomunikace a informační technologie',
          'obo': 'Obchod',
          'aut': 'automobilový průmysl',
          'ekl': 'prostředí',
          'kul': 'Kultura',
          'zdr': 'zdravotní služby',
          'vat': 'věda a technika',
          'sop': 'Sociální problémy',
          'den': 'zprávy a plány',
          'bur': 'burzy',
          'bup': 'Currency exchanges',
          'tlk': 'Telekomunikace',
          'che': 'Chemické a farmaceutické průmysl',
          'bos': 'Čeština ze zahraničí',
          'buk': 'Komoditní burzy',
          'zem': 'Zemědělství',
          'ptr': 'Potravinářský',
          'nab': 'Náboženství',
          'ene': 'Energie',
          'fot': 'fotbal',
          'bua': 'burzy akciové',
          'str': 'strojírenství',
          'bsk': 'sklářský průmysl',
          'dre': 'dřevozpracující průmysl',
          'eko': 'ekologie',
          'kat': 'nehody a katastrofy',
          'hut': 'hutnictví',
          'mot': 'motorizování',
          'pla': 'zprávy o událostech',
          'reg': 'region',
          'slo': 'slovenika ',
          'tok': 'textil',
          'zah': 'zahraniční',
          'zbr': 'zbraně',
          'mix': 'mix',
          'prm': 'lehký průmysl',
          'sur': 'suroviny',
          'pod': 'Česká republika Politika',
          }
    # di = {'<blank>': 'prázdné',
    #       '<unk>': 'neznámý',
    #       '<s>': 'začíná',
    #       '</s>': 'konec',
    #       'mak': 'makroekonomika',
    #       'slz': 'služby',
    #       'tur': 'ruch',
    #       'prg': 'Pragensie',
    #       'fin': 'finanční',
    #       'hok': 'Hokej',
    #       'eur': 'evropská',
    #       'dpr': 'Doprava',
    #       'met': 'počasí',
    #       'pol': 'Politika',
    #       'zak': 'zákon',
    #       'for': 'vlády',
    #       'sta': 'stavebnictví',
    #       'efm': 'Firmy',
    #       'spo': 'sports',
    #       'sko': 'Školství',
    #       'med': 'média',
    #       'mag': 'časopis',
    #       'spl': 'životní',
    #       'odb': 'odbory',
    #       'pit': 'informační',
    #       'obo': 'obchod',
    #       'aut': 'automobilový',
    #       'ekl': 'prostředí',
    #       'kul': 'kultura',
    #       'zdr': 'zdravotní',
    #       'vat': 'věda',
    #       'sop': 'Sociální',
    #       'den': 'plány',
    #       'bur': 'burzy',
    #       'bup': 'Currency exchanges',
    #       'tlk': 'telekomunikace',
    #       'che': 'Chemické',
    #       'bos': 'zahraničí',
    #       'buk': 'Komoditní',
    #       'zem': 'Zemědělství',
    #       'ptr': 'Potravinářský',
    #       'nab': 'Náboženství',
    #       'ene': 'Energie',
    #       'fot': 'fotbal',
    #       'bua': 'akciové',
    #       'str': 'strojírenství',
    #       }

    with open('./data/data/save_data.test.dict', 'r') as f:
        lines = f.readlines()
        labels = [line.strip().split()[0] for line in lines]

    with open('./data/data/save_data.test.dict.trans.cz', 'w', encoding='utf-8') as f:
        trans = map(lambda x: di[x] + '\n', labels)
        f.writelines(trans)

def tar_dict2json():
    '''
    构造target和test的json字典（尽管现在已经不用target了）
    :return:
    '''
    import json

    tar_dict_file = "./data/data/save_data.tgt.dict"
    test_dict_file = "./data/data/save_data.test.dict"
    # HINT: 由于标准和预测句中都不含特殊符号，所以要过滤
    start_special = 4  # 舍去的特殊字符

    with open(tar_dict_file) as f:
        target_list = f.readlines()
        tar_dict_word2index = {}
        for i in range(start_special, len(target_list)):
            word, index = target_list[i].strip().split()
            word = word + " "
            tar_dict_word2index[word] = int(index) - start_special

    with open("./data/data/target_label_dict.json", "w") as f:
        json.dump(tar_dict_word2index, f)


    with open(test_dict_file) as f:
        target_list = f.readlines()
        tar_dict_word2index = {}
        for i in range(start_special, len(target_list)):
            word, index = target_list[i].strip().split()
            word = word + " "
            tar_dict_word2index[word] = int(index) - start_special

    with open("./data/data/test_label_dict.json", "w") as f:
        json.dump(tar_dict_word2index, f)


def make_embedding(extra_word):

    with open('./data/data/save_data.test.dict.trans.cz', 'r', encoding='utf-8') as f:
        lines_test = f.readlines()

    # w2vModel = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)
    fastTextModel = FastText.load_fasttext_format('./wiki.cs.bin')

    vocab_sz = len(lines_test)
    emb_dim = 300
    weight_matrix_test = np.zeros((vocab_sz, 300))
    count = 0
    failed = []
    for i, line in enumerate(lines_test):
        label = line.strip()  # 之前还split[0]了，愚蠢啊 和dict文件的格式不一样的
        try:
            weight_matrix_test[i] = fastTextModel[label]
        except:
            count += 1
            failed.append(label)
            weight_matrix_test[i] = np.random.normal(size=(emb_dim,))
    print('failed count {}, {}'.format(count, failed))
    weight_matrix_tgt_np = torch.from_numpy(weight_matrix_test[:-1*extra_word]).float()
    weight_matrix_test_np = torch.from_numpy(weight_matrix_test).float()

    with open('./data/data/save_data.src.dict', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    vocab_sz = len(lines)
    emb_dim = 300
    weight_matrix_src = np.zeros((vocab_sz, 300))
    count = 0
    failed = []
    for i, line in enumerate(lines):
        label = line.strip().split()[0]
        try:
            weight_matrix_src[i] = fastTextModel[label]
        except:
            count += 1
            failed.append(label)
            weight_matrix_src[i] = np.random.normal(size=(emb_dim,))

    print('failed count {}, {}'.format(count, failed))

    weight_matrix_src_np = torch.from_numpy(weight_matrix_src).float()
    weight_matrix_train = {}
    weight_matrix_train['src_emb'] = weight_matrix_src_np
    weight_matrix_train['tgt_emb'] = weight_matrix_tgt_np

    weight_matrix_test = {}
    weight_matrix_test['src_emb'] = weight_matrix_src_np
    weight_matrix_test['tgt_emb'] = weight_matrix_test_np

    torch.save(weight_matrix_train, './data/data/weight_matrix_train')
    torch.save(weight_matrix_test, './data/data/weight_matrix_test')


def add_extra_word_to_testcz(new_label, disc=False):

    # 构造save_data.test.dict
    with open("./data/data/save_data.tgt.dict", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        line_num = len(lines)
        if not disc:
            for label in new_label:
                lines += label + '  ' + str(line_num) + '\n'
                line_num += 1

    with open("./data/data/save_data.test.dict", 'w', encoding='utf-8') as f:
        f.writelines(lines)


def main():
    dicts = {}
    if opt.share:
        assert opt.src_vocab_size == opt.tgt_vocab_size
        print('share the vocabulary between source and target')
        dicts['src'] = initVocabulary('source and target',
                                      [opt.train_src, opt.train_tgt],
                                      opt.src_vocab,
                                      opt.src_vocab_size)
        dicts['tgt'] = dicts['src']
    else:
        dicts['src'] = initVocabulary('source', './trg.txt', opt.src_vocab,    # './trg.txt'
                                      opt.src_vocab_size)
        dicts['tgt'] = initVocabulary('target', opt.train_tgt, opt.tgt_vocab,
                                      opt.tgt_vocab_size, char=opt.char)

    # trg_dict = makeVocabulary(opt.train_tgt, opt.tgt_vocab_size, char=opt.char, for_json=True)


    print('Preparing training ...')
    train = makeData(opt.train_src, opt.train_tgt, dicts['src'], dicts['tgt'], char=opt.char)

    print('Preparing validation ...')
    valid = makeData(opt.valid_src, opt.valid_tgt, dicts['src'], dicts['tgt'], char=opt.char)

    print('Preparing test ...')
    test = makeData(opt.test_src, opt.test_tgt, dicts['src'], dicts['tgt'], char=opt.char)

    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.tgt_vocab is None:
        saveVocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')

    print('Saving data to \'' + opt.save_data + '.train.pt\'...')
    save_data = {'dicts': dicts,
                 'train': train,  # dataset对象，包含ids句和原句等（src和trg都有）
                 'valid': valid,
                 'test': test}
    torch.save(save_data, opt.save_data)


if __name__ == "__main__":
    main()


    extra_word_num = len(new_label)

    add_extra_word_to_testcz(new_label, disc=True)

    # 构造json文件(也就是
    tar_dict2json()

    # 构造cz文件
    toTranscz()

    # 产生embedding weight matrix
    make_embedding(extra_word_num)

