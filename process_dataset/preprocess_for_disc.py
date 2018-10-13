import shutil

def filter_low_freq_labels(to_del):

    # 解放原来禁用的标签 str

    with open("trg_label.txt", 'r', encoding="utf8") as f:
        lines = f.readlines()
        labels = [line.strip().split() for line in lines]

    index_to_remove = []

    with open('trg.txt', 'r', encoding="utf8") as f:
        docs = f.readlines()

    for i, label in enumerate(labels):
        for del_item in to_del:
            if del_item in label:
                label.remove(del_item)
        if not label:
            index_to_remove.append(i)

    index_to_remove.reverse()
    for i in index_to_remove:
        del (docs[i])
        del (labels[i])

    with open('trg_filtered_frequency.txt', 'w', encoding="utf8") as f:
        f.writelines(docs)
    with open('trg_label_filtered_frequency.txt', 'w', encoding="utf8") as f:
        # re construct label
        for i, label in enumerate(labels):
            labels[i] = ' '.join(label) + '\n'
        f.writelines(labels)


def merge_label(merge_from, merge_to):
    # merge A to B
    As = ['pod']  # 'bua'
    Bs = ['pol']  # bur'

    with open('trg_label_filtered_frequency.txt', 'r' ,encoding="utf8") as f:
        lines = f.readlines()
        labels = [line.strip().split() for line in lines]

    # index_to_remove = []
    for A, B in zip(As, Bs):
        for line in labels:
            if A in line:
                if B in line:
                    line.remove(A)
                else:
                    for i in range(len(line)):
                        if line[i] == A:
                            line[i] = B

    with open('trg_label_filtered_frequency.txt', 'w' ,encoding="utf8") as f:
        # re construct label
        for i, label in enumerate(labels):
            labels[i] = ' '.join(label) + '\n'
        f.writelines(labels)


def split_test_data(test_labels):
    with open('trg_filtered_frequency.txt', 'r', encoding="utf8") as f:
        texts = f.readlines()

    with open('trg_label_filtered_frequency.txt', 'r', encoding="utf8") as f:
        label_lines = f.readlines()
        labels = []
        for i in range(len(label_lines)):
            labels.append(label_lines[i].strip().split())

    all_data = [(texts[i], labels[i]) for i in range(len(labels))]


    def isTest(labels, banned_labels):
        for ban in banned_labels:
            if ban in labels:
                return True

    def split_data(ban):
        train, test = [], []
        for data in all_data:
            if isTest(data[1], ban):
                test.append(data)
            else:
                train.append(data)
        print(len(train), len(test))
        return (train, test)

    def label_li2str(label):
        return ' '.join(label) + '\n'

    def saveToFile(train, test):
        # train
        train_doc, train_label = [], []
        for i, item in enumerate(train):
            train_doc.append(item[0])
            label = label_li2str(item[1])
            train_label.append(label)

        # test
        test_doc, test_label = [], []
        for item in test:
            test_doc.append(item[0])
            label = label_li2str(item[1])
            test_label.append(label)

        with open('train_doc.txt', 'w', encoding="utf8") as f:
            f.writelines(train_doc)
        with open('train_label.txt', 'w', encoding="utf8") as f:
            f.writelines(train_label)
        with open('./data/test_doc.txt', 'w', encoding="utf8") as f:  # test直接放到data文件夹（因为是处理好了的）
            f.writelines(test_doc)
        with open('./data/test_label.txt', 'w', encoding="utf8") as f:
            f.writelines(test_label)

    train, test = split_data(test_labels)
    saveToFile(train, test)

def split_validation_and_train(dev_size):

    with open('train_doc.txt', 'r',encoding="utf8") as f:
        doc_lines = f.readlines()

    with open('train_label.txt', 'r',encoding="utf8") as f:
        label_lines = f.readlines()

    doc_train = doc_lines[:-1*dev_size]
    doc_dev = doc_lines[-1*dev_size:]

    label_train = label_lines[:-1*dev_size]
    label_dev = label_lines[-1*dev_size:]

    with open('./data/train_doc.txt', 'w',encoding="utf8") as f:
        f.writelines(doc_train)
    with open('./data/train_label.txt', 'w',encoding="utf8") as f:
        f.writelines(label_train)
    with open('./data/dev_doc.txt', 'w',encoding="utf8") as f:
        f.writelines(doc_dev)
    with open('./data/dev_label.txt', 'w',encoding="utf8") as f:
        f.writelines(label_dev)

def merge_test_data_to_train():
    with open('./data/test_doc.txt', 'r', encoding="utf8") as f:  # test直接放到data文件夹（因为是处理好了的）
        lines = f.readlines()
        with open('./data/train_doc.txt', 'a', encoding="utf8") as ff:
            ff.writelines(lines)
    with open('./data/test_label.txt', 'r', encoding="utf8") as f:
        lines = f.readlines()
        with open('./data/train_label.txt', 'a', encoding="utf8") as ff:
            ff.writelines(lines)

def main():
    merge_from = ['pod']
    merge_to = ['pol']

    to_del = ['bsk', 'cen', 'dre', 'eko', 'kat', 'hut', 'mot',
              'pla', 'reg', 'slo', 'spc', 'tok', 'zah', 'zbr',
              'mix', 'prm', 'sur', 'str']  # note that "str" is ambiguous with aut but still use

    test_labels = ['bua', 'tlk', 'hok', 'sko', 'tur']

    dev_size = 200

    filter_low_freq_labels(to_del)

    merge_label(merge_from=merge_from, merge_to=merge_to)

    split_test_data(test_labels)

    split_validation_and_train(dev_size)

    merge_test_data_to_train()

main()
# 训练集： 包含有标签样本和标签为特殊字符样本
# 在生成train.tgt