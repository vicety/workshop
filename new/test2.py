import argparse
import torch
import data.dict as dict
from data.dataloader import dataset
from nltk import sent_tokenize
import json
from gensim.models import FastText
from gensim.models import KeyedVectors
import numpy as np

fastTextModel = FastText.load_fasttext_format('./wiki.cs.bin')

def normal2(A):
    return A / np.sqrt(np.sum(A ** 2))

while True:
    inp = input('>>>')
    try:
        A = fastTextModel[inp]
    except:
        A = np.zeros(300)
    A = normal2(A)
    inp = input('>>>')
    try:
        B = fastTextModel[inp]
    except:
        B = np.zeros(300)
    B = normal2(B)
    print(A.dot(B))