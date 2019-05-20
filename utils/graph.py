"""
Create underlying graph to represent the image
"""

import numpy as np
from scipy import sparse
from scipy.sparse import block_diag

from pygsp import utils
from pygsp.graphs import Graph


class LineGrid2d(Graph):
    def __init__(self, N1=16, N2=None, graph_orientations={}, **kwargs):

        for orientation in graph_orientations:
            if orientation not in {'left', 'right',
                                   'top', 'bottom'}:
                raise ValueError(f"{orientation} is not a valid orientation for the graph")

        if N2 is None:
            N2 = N1


        self.N1 = N1
        self.N2 = N2

        N = N1 * N2

        W = sparse.csr_matrix((N, N))

        if 'bottom' in graph_orientations:
            diag = np.ones(N - N2)
            delta = -N2
            W += sparse.diags(diagonals=[diag],
                         offsets=[delta],
                         shape=(N, N),
                         format='csr',
                         dtype='float').transpose()
        if 'top' in graph_orientations:
            diag = np.ones(N - N2)
            delta = N2
            W += sparse.diags(diagonals=[diag],
                         offsets=[delta],
                         shape=(N, N),
                         format='csr',
                         dtype='float').transpose()
        if 'left' in graph_orientations:
            diag = np.ones(N - 1)
            diag[(N2 - 1)::N2] = 0
            delta = -1
            W += sparse.diags(diagonals=[diag],
                         offsets=[delta],
                         shape=(N, N),
                         format='csr',
                         dtype='float')

        if 'right' in graph_orientations:
            diag = np.ones(N - 1)
            diag[(N2 - 1)::N2] = 0
            delta = 1
            W += sparse.diags(diagonals=[diag],
                         offsets=[delta],
                         shape=(N, N),
                         format='csr',
                         dtype='float')


        x = np.kron(np.ones((N1, 1)), (np.arange(N2)/float(N2)).reshape(N2, 1))
        y = np.kron(np.ones((N2, 1)), np.arange(N1)/float(N1)).reshape(N, 1)
        y = np.sort(y, axis=0)[::-1]
        coords = np.concatenate((x, y), axis=1)

        plotting = {"limits": np.array([-1. / N2, 1 + 1. / N2,
                                        1. / N1, 1 + 1. / N1])}

        super(LineGrid2d, self).__init__(W, coords=coords,
                                     plotting=plotting, **kwargs)

    def _get_extra_repr(self):
        return dict(N1=self.N1, N2=self.N2)


def assemble(g1, g2, margin=0.2):
    """
    Merge two graphs together
    """
    W = block_diag((g1.W, g2.W))
    margin = 0.2
    new_coords = g2.coords
    new_coords[:, 0] = new_coords[:, 0] + margin + np.max(g1.coords)
    coords = np.concatenate((g1.coords, new_coords))
    return Graph(W, coords=coords, plotting=g1.plotting)
