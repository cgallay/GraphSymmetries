# An empirical study of rotation equivariant neural networks

Charles Gallay, [Michaël Defferrard](http://deff.ch), [Nathanaël Perraudin](https://perraudin.info)

This work is a [semester project](https://www.epfl.ch/schools/ic/wp-content/uploads/2018/10/PROJETS-DE-SEMESTRE-DIRECTIVES-ENGLISH.pdf) done at [EPFL](https://www.epfl.ch/) as part of the master program of data science. 

## Goal

The goal of the project is to test different configuration in order to see the importance of being [equivariant](https://en.wikipedia.org/wiki/Equivariant_map) to rotation for a CNN.

As a first measure, classical CNN and equivariant to rotation CNN are compare on the task of image classification and more specifically on the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.  


## Setup

1. Clone this repository.
   ```sh
   git clone git@github.com:cgallay/GraphSymmetries.git
   cd GraphSymmetries
   ```

2. Install the dependencies.
   ```sh
   pip install -r requirements.txt
   pip install -e .
   ```

## Example

```python

import torch.nn as nn

from graphSym.graph_conv import GridGraphConv
from graphSym.graph_pool import GraphMaxPool2d

class Net(nn.Module):
    """
    Network with Horizontal symetry
    """
    def __init__(self, input_shape=(32,32), nb_class=5):
        super().__init__()
        underlying_graphs = [['left', 'right'], ['top'], ['bottom']]
        conv1 = GridGraphConv(3, 30, merge_way='cat', underlying_graphs=underlying_graphs)
        conv2 = GridGraphConv(30, 60, merge_way='cat', underlying_graphs=underlying_graphs)
        
        pool1 = GraphMaxPool2d(input_shape=input_shape)
        out_shape = (16, 16)
        
        conv3 = GridGraphConv(60, 60, input_shape=out_shape, merge_way='cat', underlying_graphs=underlying_graphs)
        conv4 = GridGraphConv(60, nb_class, input_shape=out_shape, merge_way='mean', underlying_graphs=underlying_graphs)
        
        self.seq = nn.Sequential(conv1, conv2, pool1, conv3, conv4)
        
    def forward(self, x):
        out = self.seq(x)
        out = out.mean(2)
        return out

net = Net()
```

## License
The content of this repository is released under the terms of the [MIT license](LICENCE.md)
