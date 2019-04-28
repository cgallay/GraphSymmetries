from typing import Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
from scipy import sparse

from models.layers.graph_conv import FixGraphConv
from models.layers.utils_graph import get_conv, get_pool
from utils.argparser import get_args
from utils.helpers import t_add, conv_output_shape
args = get_args()


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

class GraphConvNetAID(nn.Module):
    def __init__(self, input_shape=(600,600), nb_class=30):
        super(GraphConvNetAID, self).__init__()
        self.nb_class = nb_class
        layers = []

        layer, out_shape = get_layer(3, 32, input_shape, pooling_layer=True)
        layers.append(layer)

        layers.append(nn.BatchNorm1d(32))

        layer, out_shape = get_layer(32, 64, out_shape, pooling_layer=True)
        layers.append(layer)

        layers.append(nn.BatchNorm1d(64))

        layer, out_shape = get_layer(64, 64, out_shape, pooling_layer=True)
        layers.append(layer)
        
        layers.append(nn.BatchNorm1d(64))
        
        layer, out_shape = get_layer(64, 128, out_shape)
        layers.append(layer)
        
        layers.append(nn.BatchNorm1d(128))
        
        layer, out_shape = get_layer(128, 128, out_shape)
        layers.append(layer)
        
        layers.append(nn.BatchNorm1d(128))
        
        layer, out_shape = get_layer(128, 64, out_shape)
        layers.append(layer)

        layers.append(nn.BatchNorm1d(64))
        
        layer, out_shape = get_layer(64, self.nb_class, out_shape)
        layers.append(layer)

        self.seq = nn.Sequential(*layers)

        print(f"Shape before de Fully connected is {out_shape}")
        self.drop_out = nn.Dropout()
        if not args.global_average_pooling:
            print("FC layer used at the end") 
            self.fc1 = nn.Linear(self.nb_class * out_shape[0] * out_shape[1], 1000)
            self.fc2 = nn.Linear(1000, self.nb_class)

    def forward(self, x):
        out = self.seq(x)

        if not args.global_average_pooling:
            out = out.reshape(out.size(0), -1)
            out = self.drop_out(out)
            out = self.fc1(out)
            out = self.drop_out(out)
            out = self.fc2(out)
        else:
            # Global Average Pooling to agregate into 10 values.
            out = out.mean(2)
        return out
