import numpy as np
from sklearn import metrics
import torch

# # https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
# # 当一些true中的标签在perd中没有出现时，会报错  同样地：错得太离谱也会
# y_true_2 = np.array([[0, 1, 1], [1, 1, 0]])
# y_pred_2 = np.array([[1, 0, 0], [0, 0, 1]])
#
# # candidate =
#
# print (metrics.f1_score(y_true_2, y_pred_2, average='macro'))

torch.nn.Embedding.from_pretrained(
    torch.FloatTensor([1.0, 2.0])
)