import torch
import torch.utils.data as torch_data
import os
import data.utils

class dataset(torch_data.Dataset):

    def __init__(self, src, tgt, raw_src, raw_tgt, from_known):

        self.src = src
        self.tgt = tgt
        self.raw_src = raw_src
        self.raw_tgt = raw_tgt
        self.from_known = from_known

    def __getitem__(self, index):
        return self.src[index], self.tgt[index], \
               self.raw_src[index], self.raw_tgt[index], self.from_known[index]

    def __len__(self):
        return len(self.src)


def load_dataset(path):
    pass

def save_dataset(dataset, path):
    if not os.path.exists(path):
        os.mkdir(path)

# some process taken to the batch Tensor
# src, tgt -> count length of each sentence, then do padding(note that padding is id=0)
# raw_src, raw_tgt remain unprocessed
def padding(data):
    src, tgt, raw_src, raw_tgt, from_known = zip(*data)   # list [batch, length_each(not same)]

    src_len = [len(s) for s in src]  # list [batch] length of each sentence
    src_pad = torch.zeros(len(src), max(src_len)).long()  # create zero matrix enough to put src in,
                                                            #  then cast to long type for future input
    for i, s in enumerate(src):  # re-fill the sentence in
        end = src_len[i]
        src_pad[i, :end] = s[:end]

    tgt_len = [len(s) for s in tgt]
    tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
    for i, s in enumerate(tgt):
        end = tgt_len[i]
        tgt_pad[i, :end] = s[:end]

    # t -> transpose
    # return raw_src [batch, length_each]
        #    src_pad [maxlen, batch]
    return raw_src, src_pad.t(), torch.LongTensor(src_len), \
           raw_tgt, tgt_pad.t(), torch.LongTensor(tgt_len), torch.LongTensor(from_known)


def get_loader(dataset, batch_size, shuffle, num_workers):

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=padding)
    return data_loader
