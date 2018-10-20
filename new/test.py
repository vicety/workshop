import numpy as np
from sklearn import metrics
import torch

# # # https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
# # # 当一些true中的标签在perd中没有出现时，会报错  同样地：错得太离谱也会
# # y_true_2 = np.array([[0, 1, 1], [1, 1, 0]])
# # y_pred_2 = np.array([[1, 0, 0], [0, 0, 1]])
# #
# # # candidate =
# #
# # print (metrics.f1_score(y_true_2, y_pred_2, average='macro'))
#
# torch.nn.Embedding.from_pretrained(
#     torch.FloatTensor([1.0, 2.0])
# )

# import nltk
# nltk.download('reuters')

from nltk.corpus import reuters
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

n_classes = 90
labels = reuters.categories()

def load_data(config={}):
    """
    Load the Reuters dataset.

    Returns
    -------
    Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    stop_words = stopwords.words("english")
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    mlb = MultiLabelBinarizer()

    documents = reuters.fileids()
    test = [d for d in documents if d.startswith('test/')]
    train = [d for d in documents if d.startswith('training/')]

    docs = {}
    docs['train'] = [reuters.raw(doc_id) for doc_id in train]
    docs['test'] = [reuters.raw(doc_id) for doc_id in test]
    xs = {'train': [], 'test': []}
    xs['train'] = vectorizer.fit_transform(docs['train']).toarray()
    xs['test'] = vectorizer.transform(docs['test']).toarray()
    ys = {'train': [], 'test': []}
    ys['train'] = mlb.fit_transform([reuters.categories(doc_id)
                                     for doc_id in train])
    ys['test'] = mlb.transform([reuters.categories(doc_id)
                                for doc_id in test])
    data = {'x_train': xs['train'], 'y_train': ys['train'],
            'x_test': xs['test'], 'y_test': ys['test'],
            'labels': globals()["labels"]}
    return data

data = load_data()

print()

