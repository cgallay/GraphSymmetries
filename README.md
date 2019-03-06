# An empirical study of rotation equivariant neural networks

Charles Gallay, [Michaël Defferrard](http://deff.ch), [Nathanaël Perraudin](https://perraudin.info)

This work is a semester project done at [EPFL](https://www.epfl.ch/) as part of the master program of data science. 

## Goal

The goal of the project is to test different configuration in order to see the importance of being [equivariant](https://en.wikipedia.org/wiki/Equivariant_map) to rotation for a CNN.

As a first measure, classical CNN and equivariant to rotation CNN are compare on the task of image classification and more specifically on the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.  


## Installation

For a local installation, follow the below instructions.

1. Clone this repository.
   ```sh
   git clone https://github.com/SwissDataScienceCenter/DeepSphere.git
   cd DeepSphere
   ```

2. Install the dependencies.
   ```sh
   pip install -r requirements.txt
   ```

## License
The content of this repository is released under the terms of the [MIT license](LICENCE.md)
