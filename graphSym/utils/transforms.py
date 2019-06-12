
import numpy as np
import torch
import pygsp as pg
from scipy import sparse
from models.layers.graph_conv import create_laplacian

class ToGraph(object):
    def __init__(self, create_graph=False):
        self.create_graph = create_graph

    def __call__(self, x):
        # np_img = np.asarray(image)
        # x = np_img.astype(np.float32) / 255.0  # Faster computations with float32.
        # x = torch.from_numpy(x)
        # TODO: think about a better memory representation of the singal with 3 features.
        # Each feature being the color.
        
        # x = x.permute(1, 2, 0)
        signal = x.view(x.size(0), -1)

        if not self.create_graph:
            return signal
        laplacian = create_laplacian(*x.shape[:2])
        
        return (laplacian, signal)
