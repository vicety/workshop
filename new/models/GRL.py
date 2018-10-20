import torch

class GradReverse(torch.autograd.Function):
    def __init__(self, fraction):
        super(GradReverse, self).__init__()
        self.fraction = fraction

    def forward(self, x):
        return x

    def backward(self, grad_output):
        return self.fraction * (-grad_output)
