from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import pygsp as pg
from scipy import sparse

from models.layers.graph_conv import FixGraphConv


# TODO move
def create_laplacian(size_x, size_y, device='cpu'):
    graph = pg.graphs.Grid2d(size_x, size_y)
    laplacian = graph.L.astype(np.float32)
    laplacian = prepare_laplacian(laplacian)
    return laplacian.to(device)

# TODO move
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

    def scale_operator(L, lmax):
        r"""Scale an operator's eigenvalues from [0, lmax] to [-1, 1]."""
        I = sparse.identity(L.shape[0], format=L.format, dtype=L.dtype)
        L *= 2 / lmax
        L -= I
        return L

    lmax = estimate_lmax(laplacian)
    laplacian = scale_operator(laplacian, lmax)

    laplacian = sparse.coo_matrix(laplacian)

    # PyTorch wants a LongTensor (int64) as indices (it'll otherwise convert).
    indices = np.empty((2, laplacian.nnz), dtype=np.int64)
    np.stack((laplacian.row, laplacian.col), axis=0, out=indices)
    indices = torch.from_numpy(indices)

    laplacian = torch.sparse_coo_tensor(indices, laplacian.data, laplacian.shape)
    laplacian = laplacian.coalesce()  # More efficient subsequent operations.
    return laplacian


def get_conv(features_in:int, features_out:int, input_shape:Tuple[int, int]=(32, 32),
             kernel_size:int=3, padding=0, on_graph:bool=False, device:str='cpu'):
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
        On graph it is the number of border node to remove.
        On 2D Conv it is the number of zero to add on each side.
    on_graph:
        When set to True, perform a convolution on a grid graph
    device:
        For pytorch to know where to store the Laplacian matrix
    """

    if on_graph:
        kernel_size = (kernel_size // 2) + 1
        laplacian = create_laplacian(*input_shape, device=device)
        out = FixGraphConv(features_in, features_out, laplacian=laplacian,
                            kernel_size=kernel_size)
        if padding == 0:
            return out
        out = out.view(-1, input_shape)
        out = out[:, padding:-padding, padding:-padding]
        return out.view(-1, (input_shape[0] - padding) * (input_shape[1] - padding))
    else:
        return nn.Conv2d(features_in, features_out, kernel_size=kernel_size, padding=padding)


class GraphMaxPool2d(nn.Module):
    def __init__(self, kernel_size=3, stride=2, padding=1, input_shape=(32, 32)):
        super().__init__()
        self.input_shape = input_shape
        self.stride = stride
        self.pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        # TODO reshaping (view)
        x = x.permute(0,2,1)
        shape = x.shape
        x = x.view(shape[0], shape[1], self.input_shape[0], self.input_shape[1])
        x = self.pooling(x)
        x = x.view(shape[0], shape[1], shape[2] // self.stride // self.stride)
        x = x.permute(0,2,1)
        return x


def get_pool(kernel_size=3, stride=2, padding=1, input_shape=(32, 32), on_graph=False):
    if on_graph:
        return GraphMaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding,
                              input_shape=input_shape)
    else:
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)


class ConvNet(nn.Module):
    def __init__(self, input_shape=(32,32), on_graph=False, device='cpu'):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Dropout(0.2),
            get_conv(3, 96, input_shape=input_shape, kernel_size=3,
                     padding=1, on_graph=on_graph, device=device),  # out 32x32
            nn.ReLU(),
            get_conv(96, 96, input_shape=input_shape, kernel_size=3,
                     padding=1, on_graph=on_graph, device=device),  # out 32x32
            nn.ReLU(),
            get_pool(kernel_size=3, stride=2, padding=1, on_graph=on_graph)  # out 16x16 
        )
        input_shape = (input_shape[0] // 2, input_shape[1] // 2)
        self.layer2 = nn.Sequential(
            nn.Dropout(0.5),
            get_conv(96, 192, input_shape=input_shape, kernel_size=3,
                     padding=1, on_graph=on_graph, device=device),  # out 16x16
            nn.ReLU(),
            get_conv(192, 192, input_shape=input_shape, kernel_size=3,
                     padding=1, on_graph=on_graph, device=device),  # out 16x16
            nn.ReLU(),
            get_pool(kernel_size=3, stride=2, padding=1, on_graph=on_graph,
                     input_shape=input_shape)  # out 8x8
        )
        input_shape = (input_shape[0] // 2, input_shape[1] // 2)
        self.layer3 = nn.Sequential(
            nn.Dropout(0.5),
            get_conv(192, 192, input_shape=input_shape, kernel_size=3,
                     padding=1, on_graph=on_graph, device=device),  # out 16x16
            nn.ReLU(),
            get_conv(192, 192, input_shape=input_shape, kernel_size=3,
                     padding=1, on_graph=on_graph, device=device),  # out 16x16
            nn.ReLU(),
            get_pool(kernel_size=3, stride=2, padding=1, on_graph=on_graph,
                     input_shape=input_shape)  # out 8x8
        )
        input_shape = (input_shape[0] // 2, input_shape[1] // 2)
        self.layer4 = nn.Sequential(
            nn.Dropout(0.5),
            get_conv(192, 192, input_shape=input_shape, kernel_size=3,
                     padding=1, on_graph=on_graph, device=device),  # out 16x16
            nn.ReLU(),
            get_conv(192, 192, input_shape=input_shape, kernel_size=3,
                     padding=1, on_graph=on_graph, device=device),  # out 16x16
            nn.ReLU(),
            get_pool(kernel_size=3, stride=2, padding=1, on_graph=on_graph,
                     input_shape=input_shape)  # out 8x8
        )
        input_shape = (input_shape[0] // 2, input_shape[1] // 2)
        self.layer5 = nn.Sequential(
            nn.Dropout(0.5),
            get_conv(192, 192, input_shape=input_shape, kernel_size=3,
                     padding=1, on_graph=on_graph, device=device),  # out 8x8
            nn.ReLU(),
            get_conv(192, 10, input_shape=input_shape, device=device, kernel_size=1,
                     on_graph=on_graph),  # out 8x8
            nn.ReLU()
        )
        
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(10 * input_shape[0] * input_shape[1], 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop_out(out)
        out = self.fc2(out)
        return out
