from typing import Tuple

import torch
import torch.nn as nn
from scipy import sparse

from models.layers.graph_conv import FixGraphConv
from utils.argparser import get_args
from utils.helpers import t_add, conv_output_shape
args = get_args()

def get_conv(features_in:int, features_out:int, input_shape:Tuple[int, int]=(32, 32),
             kernel_size:int=3, padding=0, on_graph:bool=False, crop_size:int=0,
             device:str='cpu'):
    """
    Define a convolution either on Graph or a normal 2DConv depending on parameter on_graph.

    When then convolution is defined on graph, a laplacian for a grid graph is created and used
    to perform the convolution.
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
        Size of the square kernel for 2DConv or number of chebyshev polynom 
        to use for aproximation. It can be seen as the further filter can see
        in term of hope.
    padding:
        The number of zero to add on each side.
    on_graph:
        When set to True, perform a convolution on a grid graph
    remove_boundary_effect:
        When True the K (kernel_size) nodes on the border of the graph are removed in order to
        compensate for the reflection effect on the boundary.
    device:
        For pytorch to know where to store the Laplacian matrix
    """

    if on_graph:
        conv = FixGraphConv(features_in, features_out, input_shape=input_shape,
                           kernel_size=kernel_size, padding=padding, device=device,
                           crop_size=crop_size)
        out_shape = t_add(input_shape, padding - crop_size)
        return conv, out_shape
    else:
        conv = nn.Conv2d(features_in, features_out, kernel_size=kernel_size, padding=padding)
        out_shape = conv_output_shape(input_shape, kernel_size=kernel_size, pad=padding)
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


def get_pool(kernel_size=3, stride=2, padding=1, input_shape=(32, 32), on_graph=False):
    out_shape = conv_output_shape(input_shape, kernel_size, stride, pad=padding)
    if on_graph:
        pool =  GraphMaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding,
                               input_shape=input_shape)
    else:
        pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    return pool, out_shape


def get_layer(nb_channel_in, nb_channel_out, input_shape):
    conv, out_shape = get_conv(nb_channel_in, nb_channel_out, input_shape=input_shape,
                               kernel_size=5, padding=0, on_graph=True, device=args.device,
                               crop_size=0)

    pool, out_shape = get_pool(kernel_size=3, stride=2, padding=1, on_graph=True,
                               input_shape=out_shape)

    layer = nn.Sequential(
                nn.Dropout(0.2),
                conv,
                nn.ReLU(),
                pool
            )
    return layer, out_shape

class GraphConvNet(nn.Module):
    def __init__(self, input_shape=(32,32), on_graph=True, device='cpu',
                 nb_class:int=10):
        super(GraphConvNet, self).__init__()
        self.nb_class = nb_class
        conv1, shape1 = get_conv(3, 96, input_shape=input_shape, kernel_size=5,
                               padding=1, on_graph=True, device=device,
                               crop_size=1)  # out 32x32
        pool1, pshape1 = get_pool(kernel_size=3, stride=2, padding=1, on_graph=True,
                                  input_shape=shape1)  # out 16x16
        self.layer1, out_shape = get_layer(3, 96, input_shape)
        self.layer2, out_shape = get_layer(96, 192, out_shape)
        self.layer3, out_shape = get_layer(192, 192, out_shape)
        self.layer4, out_shape = get_layer(192, self.nb_class, out_shape)
        

        print(f"Shape before de Fully connected is {out_shape}")
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(self.nb_class * out_shape[0] * out_shape[1], 1000)
        self.fc2 = nn.Linear(1000, self.nb_class)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if args.fully_connected:
            out = out.reshape(out.size(0), -1)
            out = self.fc1(out)
            out = self.drop_out(out)
            out = self.fc2(out)
        else:
            # Global Average Pooling to agregate into 10 values.
            out = out.mean()
        return out
