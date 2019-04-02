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
             kernel_size:int=3, padding=0):
    conv = nn.Conv2d(features_in, features_out, kernel_size=kernel_size, padding=padding)
    out_shape = conv_output_shape(input_shape, kernel_size=kernel_size, pad=padding)
    return conv, out_shape

def get_pool(kernel_size=3, stride=2, padding=1, input_shape=(32, 32)):
    out_shape = conv_output_shape(input_shape, kernel_size, stride, pad=padding)
    pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    return pool, out_shape


def get_layer(nb_channel_in, nb_channel_out, input_shape, pooling_layer=True, dropout_rate=0.5):
    conv, out_shape = get_conv(nb_channel_in, nb_channel_out, input_shape=input_shape,
                               kernel_size=3, padding=1)
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


class ConvNet(nn.Module):
    def __init__(self, input_shape=(32,32), nb_class:int=10):
        super(ConvNet, self).__init__()
        self.nb_class = nb_class
        layers = []

        layer, out_shape = get_layer(3, 96, input_shape, dropout_rate=0.2,
                                     pooling_layer=False)
        layers.append(layer)
        layer, out_shape = get_layer(96, 96, out_shape, dropout_rate=0.0)
        layers.append(layer)


        layer, out_shape = get_layer(96, 192, out_shape, dropout_rate=0.5,
                                     pooling_layer=False)
        layers.append(layer)
        layer, out_shape = get_layer(192, 192, out_shape, dropout_rate=0.0)
        layers.append(layer)


        layer, out_shape = get_layer(192, 192, out_shape, dropout_rate=0.5,
                                     pooling_layer=False)
        layers.append(layer)
        layer, out_shape = get_layer(192, 192, out_shape, dropout_rate=0.0)
        layers.append(layer)


        layer, out_shape = get_layer(192, 192, out_shape, dropout_rate=0.5,
                                     pooling_layer=False)
        layers.append(layer)
        layer, out_shape = get_layer(192, self.nb_class, out_shape, dropout_rate=0.0)
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
