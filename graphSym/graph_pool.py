import torch.nn as nn
from graphSym.utils.helpers import conv_output_shape


class GraphMaxPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0, input_shape=(32, 32)):
        super().__init__()
        self.input_shape = input_shape
        self.stride = stride
        self.out_shape = conv_output_shape(input_shape, kernel_size, stride, padding=padding)
        self.pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # TODO reshaping (view)
        shape = x.shape
        x = x.view(shape[0], shape[1], self.input_shape[0], self.input_shape[1])
        x = self.pooling(x)
        x = x.view(shape[0], shape[1], self.out_shape[0] * self.out_shape[1])
        return x