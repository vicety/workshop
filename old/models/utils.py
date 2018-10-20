import torch

def l2norm(x):
    return x.div(torch.norm(x, 2, 1).unsqueeze(1))