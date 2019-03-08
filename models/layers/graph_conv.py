#!/usr/bin/env python3

r"""
PyTorch implementation of a convolutional neural network on graphs based on
Chebyshev polynomials of the graph Laplacian.
See https://arxiv.org/abs/1606.09375 for details.
Copyright 2018 Michaël Defferrard.
Released under the terms of the MIT license.
"""


import math

from scipy import sparse
import torch


# State-less function.
def graph_conv(laplacian, x, weight):
    B, V, Fin = x.shape
    Fin, K, Fout = weight.shape
    # B = batch size
    # V = nb vertices
    # Fin = nb input features
    # Fout = nb output features
    # K = order of Chebyshev polynomials (kenel size)

    # transform to Chebyshev basis
    x0 = x.permute(1, 2, 0).contiguous()  # V x Fin x B
    x0 = x0.view([V, Fin*B])              # V x Fin*B
    x = x0.unsqueeze(0)                   # 1 x V x Fin*B

    if K > 1:
        x1 = torch.sparse.mm(laplacian, x0)     # V x Fin*B
        x = torch.cat((x, x1.unsqueeze(0)), 0)  # 2 x V x Fin*B
    for _ in range(2, K):
        x2 = 2 * torch.sparse.mm(laplacian, x1) - x0
        x = torch.cat((x, x2.unsqueeze(0)), 0)  # M x Fin*B
        x0, x1 = x1, x2

    x = x.view([K, V, Fin, B])              # K x V x Fin x B
    x = x.permute(3, 1, 2, 0).contiguous()  # B x V x Fin x K
    x = x.view([B*V, Fin*K])                # B*V x Fin*K

    # Linearly compose Fin features to get Fout features
    weight = weight.view(Fin*K, Fout)
    x = x.matmul(weight)      # B*V x Fout
    x = x.view([B, V, Fout])  # B x V x Fout

    return x


class FixGraphConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, laplacian, kernel_size=3, bias=True,
                 conv=graph_conv):
        super().__init__()
        self.laplacian = laplacian
        self.conv = GraphConv(in_channels, out_channels, kernel_size=kernel_size,
                              bias=bias, conv=graph_conv)
    
    def forward(self, x):
        return self.conv.forward(self.laplacian, x)


# State-full class.
class GraphConv(torch.nn.Module):
    """Graph convolutional layer.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input graph.
    out_channels : int
        Number of channels in the output graph.
    kernel_size : int
        Size of the convolving kernel in number of hops.
        It corresponds to the order of the Chebyshev polynomials.
        A kernel_size of 0 won't take the neighborhood into account.
        A kernel_size of 1 will look at the 1-neighborhood.
        A kernel_size of 2 will look at the 1- and 2-neighborhood.
    bias : bool
        Whether to add a bias term.
    conv : callable
        Function which will perform the actual convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True,
                 conv=graph_conv):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._conv = conv

        # shape = (kernel_size, out_channels, in_channels)
        shape = (in_channels, kernel_size, out_channels)
        self.weight = torch.nn.Parameter(torch.Tensor(*shape))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Reset the coefficients and biases to random normal values."""
        std = 1 / math.sqrt(self.in_channels * self.kernel_size)
        self.weight.data.normal_(0, std)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def set_parameters(self, weight, bias=None):
        r"""Set weight and bias.

        Parameters
        ----------
        weight : array of shape in_channels x kernel_size x out_channels
            The coefficients of the Chebyshev polynomials.
        bias : vector of length out_channels
            The bias.
        """
        self.weight = torch.nn.Parameter(torch.as_tensor(weight))
        if bias is not None:
            self.bias = torch.nn.Parameter(torch.as_tensor(bias))
        else:
            self.register_parameter('bias', None)

    def extra_repr(self):
        s = '{in_channels} -> {out_channels}, kernel_size={kernel_size}'
        s += ', bias=' + str(self.bias is not None)
        return s.format(**self.__dict__)

    def forward(self, laplacian, inputs):
        r"""Forward graph convolution.

        Parameters
        ----------
        laplacian : sparse matrix of shape n_vertices x n_vertices
            Encode the graph structure.
        inputs : tensor of shape n_signals x n_vertices x n_features
            Data, i.e., features on the vertices.
        """
        outputs = self._conv(laplacian, inputs, self.weight)
        if self.bias is not None:
            outputs += self.bias
        return outputs
