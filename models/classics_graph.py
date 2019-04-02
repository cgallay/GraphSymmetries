from typing import Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
from scipy import sparse

from models.layers.graph_conv import FixGraphConv
from utils.argparser import get_args
from utils.helpers import t_add, conv_output_shape
args = get_args()

def get_conv(features_in:int, features_out:int, input_shape:Tuple[int, int]=(32, 32),
             kernel_size:int=3, padding=0, crop_size:int=0):
    """
    Define a convolution on Graph

    The laplacian for a grid graph is created and used to perform the convolution.

    In case of padding we construct a noramal graph but from the output graph, we remove the verticies
    that are on the side. (If a verticies has less than 4 neibough or the max number of neigbour it is removed
    from the graph.)  

    Param:
    features_in:
        number of channel in input of the convolution
    features_out:
        Number of channel in output of the convolution
    input_shape:
        Shape of the input image (usually a square tuple)
    kernel_size:
        number of chebyshev polynom to use for aproximation.
        It can be seen as the further filter can see in term of hope.
    padding:
        The number of zero to add on each side.
    crop_size:
        Number of node to remove from the grid graph (on the border).
        Idealy it should be equal to K in order to remove border effect.
    """


    conv = FixGraphConv(features_in, features_out, input_shape=input_shape,
                        kernel_size=kernel_size, padding=padding,
                        crop_size=crop_size)
    out_shape = t_add(input_shape, padding - crop_size)
    return conv, out_shape

def crop(img, s):
    return img[:, s:-s, s:-s]

class GraphMaxPool2d(nn.Module):
    def __init__(self, kernel_size=3, stride=2, padding=1, input_shape=(32, 32)):
        super().__init__()
        self.input_shape = input_shape
        self.stride = stride
        self.pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        # TODO reshaping (view)
        shape = x.shape
        x = x.view(shape[0], shape[1], self.input_shape[0], self.input_shape[1])
        x = self.pooling(x)
        x = x.view(shape[0], shape[1], shape[2] // self.stride // self.stride)
        return x


def get_pool(kernel_size=3, stride=2, padding=1, input_shape=(32, 32)):
    out_shape = conv_output_shape(input_shape, kernel_size, stride, pad=padding)

    pool = GraphMaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding,
                          input_shape=input_shape)
    return pool, out_shape


def get_layer(nb_channel_in, nb_channel_out, input_shape, pooling_layer=True, dropout_rate=0.0):
    conv, out_shape = get_conv(nb_channel_in, nb_channel_out, input_shape=input_shape,
                               kernel_size=5, padding=0, crop_size=0)
    seq = OrderedDict()
    if dropout_rate > 0 :
        seq['dropout'] = nn.Dropout(dropout_rate)
    seq['conv'] = conv
    seq['relu'] = nn.ReLU()
    if pooling_layer:
        pool, out_shape = get_pool(kernel_size=3, stride=2, padding=1,
                                   input_shape=out_shape)
        seq['pool'] = pool

    layer = nn.Sequential(seq)
    return layer, out_shape

class GraphConvNet(nn.Module):
    def __init__(self, input_shape=(32,32), nb_class:int=10):
        super(GraphConvNet, self).__init__()
        self.nb_class = nb_class
        layers = []

        layer, out_shape = get_layer(3, 192, input_shape, pooling_layer=True)
        layers.append(layer)


        layer, out_shape = get_layer(192, 384, out_shape, pooling_layer=True)
        layers.append(layer)


        layer, out_shape = get_layer(384, 384, out_shape, pooling_layer=True)
        layers.append(layer)


        layer, out_shape = get_layer(384, self.nb_class, out_shape)
        layers.append(layer)

        self.seq = nn.Sequential(*layers)

        print(f"Shape before de Fully connected is {out_shape}")
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(self.nb_class * out_shape[0] * out_shape[1], 1000)
        self.fc2 = nn.Linear(1000, self.nb_class)

    def forward(self, x):
        out = self.seq(x)

        if args.fully_connected:
            out = out.reshape(out.size(0), -1)
            out = self.fc1(out)
            out = self.drop_out(out)
            out = self.fc2(out)
        else:
            # Global Average Pooling to agregate into 10 values.
            out = out.mean()
        return out
