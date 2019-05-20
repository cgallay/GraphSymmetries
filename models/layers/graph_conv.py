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
from torch import nn
import torch.nn.functional as F
import numpy as np
import pygsp as pg

from grid2d import Grid2d
from utils.argparser import get_args
from utils.graph import LineGrid2d
args = get_args()


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


def create_laplacian(size_x, size_y):
    diagonal = 1/math.pow(2, 0.5) if args.diagonals else 0.0 
    graph = Grid2d(size_x, size_y, diagonal=diagonal)
    laplacian = graph.L.astype(np.float32)
    laplacian = prepare_laplacian(laplacian)
    return laplacian.to(args.device)


def create_vertical_laplacian(size_x, size_y, vertical=True):
    diagonal = 1/math.pow(2, 0.5) if args.diagonals else 0.0 
    graph = LineGrid2d(size_x, size_y, {'top', 'bottom'} if vertical else {'left', 'right'})
    laplacian = graph.L.astype(np.float32)
    laplacian = prepare_laplacian(laplacian)
    return laplacian.to(args.device)

def create_random_walk_matrix(size_x, size_y, graph_orientations={}):
    """
    differents modes:

    0: use the grid2d graph
    """
    graph = LineGrid2d(size_x, size_y, graph_orientations)
    rand_walk = np.diag(np.ones(graph.d.shape) / graph.d) @ graph.A
    rand_walk = rand_walk.astype(np.float32)

    # Convert to pytorch sparse tensor
    rand_walk = sparse.coo_matrix(rand_walk)

    # PyTorch wants a LongTensor (int64) as indices (it'll otherwise convert).
    indices = np.empty((2, rand_walk.nnz), dtype=np.int64)
    np.stack((rand_walk.row, rand_walk.col), axis=0, out=indices)
    indices = torch.from_numpy(indices)

    rand_walk = torch.sparse_coo_tensor(indices, rand_walk.data, rand_walk.shape)
    rand_walk = rand_walk.coalesce()  # More efficient subsequent operations
    return rand_walk.to(args.device)


def prepare_laplacian(laplacian):
    r"""Prepare a graph Laplacian to be fed to a graph convolutional layer."""

    def estimate_lmax(laplacian, tol=5e-3):
        r"""Estimate the largest eigenvalue of an operator."""
        lmax = sparse.linalg.eigsh(laplacian, k=1, tol=tol,
                                   ncv=min(laplacian.shape[0], 10),
                                   return_eigenvectors=False)
        lmax = lmax[0]
        lmax *= 1 + 2*tol  # Be robust to errors.
        return lmax

    def scale_operator(L, lmax, scale=1):
        r"""Scale an operator's eigenvalues from [0, lmax] to [-scale, scale]."""
        I = sparse.identity(L.shape[0], format=L.format, dtype=L.dtype)
        L *= 2 * scale / lmax
        L -= I
        return L

    lmax = estimate_lmax(laplacian)
    laplacian = scale_operator(laplacian, lmax, args.L_scale)

    laplacian = sparse.coo_matrix(laplacian)

    # PyTorch wants a LongTensor (int64) as indices (it'll otherwise convert).
    indices = np.empty((2, laplacian.nnz), dtype=np.int64)
    np.stack((laplacian.row, laplacian.col), axis=0, out=indices)
    indices = torch.from_numpy(indices)

    laplacian = torch.sparse_coo_tensor(indices, laplacian.data, laplacian.shape)
    laplacian = laplacian.coalesce()  # More efficient subsequent operations.
    return laplacian

class FixGraphConv(torch.nn.Module):
    """
    Convolution on a Grid graph. For images.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True,
                 input_shape=(32,32), conv=graph_conv, padding=0, crop_size=0, graph_pooling=False):
        super().__init__()
        if not graph_pooling and args.vertical_graph:
            if not out_channels % 2 == 0:
                raise InputError("When no pooling over the subgraph is applyed the number of out feature should be even.")
            out_channels = out_channels // 2

        self.new_input_shape = (input_shape[0] + 2 * padding,
                                input_shape[1] + 2 * padding)

        # Create the random walk matrix for each sub-graph
        rand_walks = []
        for graph in args.graphs:
            print(graph)
            rand_walks.append(create_random_walk_matrix(*self.new_input_shape, graph))

        if args.same_filters:
            self.conv = GraphConv(in_channels, out_channels, kernel_size=kernel_size,
                                  bias=bias, conv=graph_conv)
        else:
            convs = []
            for i in range(len(args.graphs)):
                convs.append(GraphConv(in_channels, out_channels, kernel_size=kernel_size,
                                  bias=bias, conv=graph_conv))
            self.convs = nn.ModuleList(convs)

        self.padding = padding
        self.input_shape = input_shape
        self.crop_size = crop_size
        self.graph_pooling = graph_pooling  # when set to true, the features of subgraph are pooled together with mean pooling
    def _pad(self, x):
        if self.padding > 0:
            x = x.view(x.size(0), x.size(1), *self.input_shape)
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
            x = x.view(x.size(0), x.size(1), -1)
        return x

    def _crop(self, x):
        s = self.crop_size
        if s > 0:
            x = x.view(x.size(0), x.size(1), *self.new_input_shape)
            x = x[:, :, s:-s, s:-s]
            x = x.contiguous()
            x = x.view(x.size(0), x.size(1), -1)
        return x

    def forward(self, x):
        """
        x is of shape [batch_size, nb_node, nb_features_in]
        """
        x = self._pad(x)
        x = x.permute(0, 2, 1)  # shape it for the graph conv

        # Compute outputs
        outs = []
        for i, rand_walk in enumerate(self.convs):
            if args.same_filters:
                conv = self.conv
            else:
                conv = self.convs[i]
            outs.append(conv(rand_walk, x)


        # Merge output
        if self.is_last_layer or args.way_of_merge == 'mean':
            x = torch.stack(outs).mean(0)
        else:
            if args.way_of_merge == 'concat':
                # flat all the output for a node
                x = torch.cat(outs, -1)

        x = x.permute(0, 2, 1).contiguous()  # reshape as before
        x = self._crop(x)
        return x


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
        std = 1 / math.sqrt(self.in_channels * (self.kernel_size + 0.5) / 2)
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
