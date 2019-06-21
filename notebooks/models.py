import torch.nn as nn

from graphSym.graph_conv import GridGraphConv
from graphSym.graph_pool import GraphMaxPool2d


class CIFARNet(nn.Module):
    """
    Network for cifar
    """
    def __init__(self, input_shape=(32, 32), nb_class=10, underlying_graphs=None):
        super().__init__()
        if underlying_graphs is None:
            underlying_graphs = [['left', 'right'], ['top'], ['bottom']]
            
        nb_features = 30
        
        conv1 = GridGraphConv(3, nb_features, merge_way='cat', underlying_graphs=underlying_graphs)
        conv2 = GridGraphConv(nb_features, nb_features, merge_way='cat', underlying_graphs=underlying_graphs)
        pool1 = GraphMaxPool2d(input_shape=input_shape)
        out_shape = pool1.out_shape  # (16, 16)

        conv3 = GridGraphConv(nb_features, nb_features, input_shape=out_shape, merge_way='cat', underlying_graphs=underlying_graphs)
        conv4 = GridGraphConv(nb_features, nb_features, input_shape=out_shape, merge_way='cat', underlying_graphs=underlying_graphs)
        pool2 = GraphMaxPool2d(input_shape=out_shape)
        out_shape = pool2.out_shape  # (8, 8)
        
        conv5 = GridGraphConv(nb_features, nb_features, input_shape=out_shape, merge_way='cat', underlying_graphs=underlying_graphs)
        conv6 = GridGraphConv(nb_features, nb_features, input_shape=out_shape, merge_way='cat', underlying_graphs=underlying_graphs)
        pool3 = GraphMaxPool2d(input_shape=out_shape)
        out_shape = pool3.out_shape  # (4, 4)

        conv7 = GridGraphConv(nb_features, nb_features, input_shape=out_shape, merge_way='cat', underlying_graphs=underlying_graphs)
        conv8 = GridGraphConv(nb_features, nb_features, input_shape=out_shape, merge_way='cat', underlying_graphs=underlying_graphs)
        pool4 = GraphMaxPool2d(input_shape=out_shape)
        out_shape = pool4.out_shape  # (2, 2)
        
        conv9 = GridGraphConv(nb_features, nb_features, input_shape=out_shape, merge_way='cat', underlying_graphs=underlying_graphs)
        conv10 = GridGraphConv(nb_features, nb_class, input_shape=out_shape, merge_way='mean', underlying_graphs=underlying_graphs)

        self.seq = nn.Sequential(conv1, conv2, pool1,
                                 conv3, conv4, pool2,
                                 conv5, conv6, pool3,
                                 conv7, conv8, pool4,
                                 conv9, conv10)
        
    def forward(self, x):
        out = self.seq(x)
        out = out.mean(2)
        return out

class AIDNet(nn.Module):
    """
    Network for cifar
    """
    def __init__(self, input_shape=(200, 200), nb_class=30, underlying_graphs=None):
        super().__init__()
        if underlying_graphs is None:
            underlying_graphs = [['left', 'right'], ['top'], ['bottom']]
            
        nb_features = 30
        
        conv1 = GridGraphConv(3, nb_features, input_shape=input_shape, merge_way='cat', underlying_graphs=underlying_graphs)
        conv2 = GridGraphConv(nb_features, nb_features, input_shape=input_shape, merge_way='cat', underlying_graphs=underlying_graphs)
        pool1 = GraphMaxPool2d(input_shape=input_shape)
        out_shape = pool1.out_shape  # (100, 100)

        conv3 = GridGraphConv(nb_features, nb_features, input_shape=out_shape, merge_way='cat', underlying_graphs=underlying_graphs)
        conv4 = GridGraphConv(nb_features, nb_features, input_shape=out_shape, merge_way='cat', underlying_graphs=underlying_graphs)
        pool2 = GraphMaxPool2d(input_shape=out_shape)
        out_shape = pool2.out_shape  # (50, 50)
        
        conv5 = GridGraphConv(nb_features, nb_features, input_shape=out_shape, merge_way='cat', underlying_graphs=underlying_graphs)
        conv6 = GridGraphConv(nb_features, nb_features, input_shape=out_shape, merge_way='cat', underlying_graphs=underlying_graphs)
        pool3 = GraphMaxPool2d(input_shape=out_shape, padding=1)
        out_shape = pool3.out_shape  # (26, 26)

        conv7 = GridGraphConv(nb_features, nb_features, input_shape=out_shape, merge_way='cat', underlying_graphs=underlying_graphs)
        conv8 = GridGraphConv(nb_features, nb_features, input_shape=out_shape, merge_way='cat', underlying_graphs=underlying_graphs)
        pool4 = GraphMaxPool2d(input_shape=out_shape, padding=1)
        out_shape = pool4.out_shape  # (14, 14)
        
        conv9 = GridGraphConv(nb_features, nb_features, input_shape=out_shape, merge_way='cat', underlying_graphs=underlying_graphs)
        conv10 = GridGraphConv(nb_features, nb_features, input_shape=out_shape, merge_way='cat', underlying_graphs=underlying_graphs)
        pool5 = GraphMaxPool2d(input_shape=out_shape, padding=1)
        out_shape = pool5.out_shape  # (8, 8)
        
        conv11 = GridGraphConv(nb_features, nb_features, input_shape=out_shape, merge_way='cat', underlying_graphs=underlying_graphs)
        conv12 = GridGraphConv(nb_features, nb_features, input_shape=out_shape, merge_way='cat', underlying_graphs=underlying_graphs)
        pool6 = GraphMaxPool2d(input_shape=out_shape)
        out_shape = pool6.out_shape  # (4, 4)
        
        conv13 = GridGraphConv(nb_features, nb_features, input_shape=out_shape, merge_way='cat', underlying_graphs=underlying_graphs)
        conv14 = GridGraphConv(nb_features, nb_class, input_shape=out_shape, merge_way='mean', underlying_graphs=underlying_graphs)

        self.seq = nn.Sequential(conv1, conv2, pool1,
                                 conv3, conv4, pool2,
                                 conv5, conv6, pool3,
                                 conv7, conv8, pool4,
                                 conv9, conv10, pool5,
                                 conv11, conv12, pool6,
                                 conv13, conv14)
        
    def forward(self, x):
        out = self.seq(x)
        out = out.mean(2)
        return out
    
