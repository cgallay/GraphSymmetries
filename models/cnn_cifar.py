from collections import OrderedDict

import torch.nn as nn
from models.layers.utils_cnn import get_conv, get_pool

from utils.argparser import get_args
args = get_args()

class CNNConvNetCIFAR(nn.Module):
    def __init__(self, input_shape=(32, 32), nb_class=10):
        super(CNNConvNetCIFAR, self).__init__()
        self.nb_class = nb_class
        layers = []
        
        layer, out_shape = get_layer(3, 80, input_shape, pooling_layer=False)
        layers.append(layer)
        layer, out_shape = get_layer(80, 80, out_shape)
        layers.append(layer)

        layers.append(nn.BatchNorm2d(80))

        layer, out_shape = get_layer(80, 140, out_shape, pooling_layer=False)
        layers.append(layer)
        layer, out_shape = get_layer(140, 140, out_shape)
        layers.append(layer)

        layers.append(nn.BatchNorm2d(140))

        layer, out_shape = get_layer(140, 140, out_shape, pooling_layer=False)
        layers.append(layer)
        layer, out_shape = get_layer(140, 140, out_shape)
        layers.append(layer)

        layers.append(nn.BatchNorm2d(140))

        layer, out_shape = get_layer(140, 140, out_shape, pooling_layer=False)
        layers.append(layer)
        layer, out_shape = get_layer(140, self.nb_class, out_shape)
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
            out = self.drop_out(out)
            out = self.fc1(out)
            out = self.drop_out(out)
            out = self.fc2(out)
        else:
            # Global Average Pooling to agregate into 10 values.
            out = out.mean()
        return out

def get_layer(nb_channel_in, nb_channel_out, input_shape, pooling_layer=True, dropout_rate=0.0):
    conv, out_shape = get_conv(in_channels=nb_channel_in, out_channels=nb_channel_out, input_shape=input_shape,
                               kernel_size=3, padding=1, crop_size=0)
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

