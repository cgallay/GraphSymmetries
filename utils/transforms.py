
import numpy as np
import torch
import pygsp as pg
from scipy import sparse

class ToGraph(object):
    def __init__(self, create_graph=False):
        self.create_graph = create_graph

    def __call__(self, image):
        np_img = np.asarray(image)
        x = np_img.astype(np.float32)  # Faster computations with float32.
        x = torch.from_numpy(x)
        # TODO: think about a better memory representation of the singal with 3 features.
        # Each feature being the color.
        signal = x.view(-1, 3)

        if not self.create_graph:
            return signal
        laplacian = create_laplacian(*np_img.shape[:2])
        
        return (laplacian, signal)


def create_laplacian(size_x, size_y):
    graph = pg.graphs.Grid2d(size_x, size_y)
    laplacian = graph.L.astype(np.float32)
    laplacian = prepare_laplacian(laplacian)
    return laplacian


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


