import torch.nn as nn

from models.layers.graph_conv import FixGraphConv
from utils.helpers import t_add, conv_output_shape


def get_pool(kernel_size=3, stride=2, padding=1, input_shape=(32, 32)):
    out_shape = conv_output_shape(input_shape, kernel_size, stride, padding=padding)

    pool = GraphMaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding,
                          input_shape=input_shape)
    return pool, out_shape


def get_conv(features_in, features_out, input_shape=(32, 32),
             kernel_size=3, merge_way='mean', same_filters=True,
             underlying_graphs=None):
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
    """
    if underlying_graphs is None:
        underlying_graphs = [{'left', 'right', 'top', 'bottom'}]

    conv = FixGraphConv(features_in, features_out, input_shape=input_shape,
                        kernel_size=kernel_size,
                        merge_way=merge_way,
                        same_filters=same_filters,
                        underlying_graphs=underlying_graphs)
    return conv, input_shape

class GraphMaxPool2d(nn.Module):
    def __init__(self, kernel_size=3, stride=2, padding=1, input_shape=(32, 32)):
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
