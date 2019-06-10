---
title:  "How to choose the image’s symmetries with graph neural network"
layout: splash
classes: wide

excerpt: "[Charles Gallay](https://github.com/cgallay), [Michaël Defferrard](http://deff.ch) and [Nathanaël Perraudin](https://perraudin.info)"
header:
  overlay_image: docs/images/symmetries.png
  overlay_filter: rgba(125, 125, 125, 0.9)
  caption: "Photo credit: [**Wikipedia**](https://en.wikipedia.org/wiki/Symmetry_group)"


feature_row:
  - image_path: docs/images/2dGrid.png
    alt: "placeholder image 1"
    title: # "Placeholder 1"
    excerpt: "First graph, the 2dGrid graph is the simplest regular graph defined on the plane. It's undirected edges gives him multiple symmetries. The ones we are interested in are: rotation of 90 degrees and mirroring."

  - image_path: docs/images/vertical_horizontal.png
    alt: "placeholder image 2"
    title: # "Placeholder 2"
    excerpt: "The second kind of graphs are undirected vertical or horizontal edges. The symmetries found here are vertical and horizontal flips."

  - image_path: docs/images/all_direction.png
    title: # "Placeholder 3"
    excerpt: "Last type of graphs are simple directed graph. Those graph doesn’t contain any symmetry but allow the information to flow across the image when combined with other graphs."

---



## Motivation:

Symmetries are naturally present in images. From two similar objects appearing in different locations to different point of view. This high level of symmetries induce a high degree of correlation between pixels that is worth exploiting. In this blogpost provide the reader with a [framework](https://github.com/cgallay/GraphSymmetries/) that aim at testing hidden symmetries of datasets. we showed that exploiting the true symmetry does infact boost performances. Once again showing that knowing your dataset is something that still has a huge impact.
{: .text-justify}


## Exploiting Symmetries

To exploit those symmetries people have apply transformations to the input image. Newly created synthetic images have the property of still belonging to the original class. While those images might seem obviously the same for humans, for CNN they appear as quite different ones. This allows the network to learn news features extractor and perform better on images close to transformed one. Depending on the dataset typical transformation include but are not limited to  random rotations,cropping, illuminance variation and resizing. This technique called “Data Augmentation” has been shown to help CNN to better generalization.
{: .text-justify}


But creating those artificial data introduce a lot of correlation into the wights learned during training. What if you could exploit those symmetry in a smarter way and build them directly into the network ? What if the network you design could react in a predictable way when faced with transformed data? This property is what we call equivariance. In the case of CNN equivariance to translation allows weight sharing and has been shown to help a lot. Recently [Cohen](https://github.com/tscohen/GrouPy), has shown that you can design network equivariant to any transformation from any group. In his implementation the weights of the filter are shared with 90 degree rotations versions of the input. As an illustration of the utility, the network will have to learn only one edge detector, while a standard CNN would have to learn two vertical and two horizontal ones.
{: .text-justify}

> “Convolutional structure is not just a sufficient, but
also a necessary condition for equivariance to the
action of a compact group.” - [Risi and Shubhendu](https://arxiv.org/abs/1802.03690)




## Grid Graph symmetries

### Underlying graphs

To exploit the symmetries, we works on graph where convolution is invariant to permutation of neighbouring nodes. Network we design will have invariance property to different transformation depending on the underlying graph we use. We present here the different underlying graph we used and explain their group symmetry.
{: .text-justify}

{% include feature_row %}


By concatenation the output of the Graph Convolution on those different underlying graph, we can build network that are invariant to desire symmetries only. For example in the case of the graph below, we have a GCNN that is only invariant to horizontal mirroring.
{: .text-justify}



Different combination of those graph allows the convolution to be invariant to different symmetries.
{: .text-justify}


## Experimentation

| Symmeties           | CIFAR-10 | AID |
|---------------------|----------|-----|
| All (2dGrid)        |          |     |
| Vertical+Horizontal |          |     |
| Vertical            |          |     |
| Horizontal          |          |     |
| None (Directed)     |          |     |

### Dataset
Coming from all that background, we want to check which symeties is worth exploiting. We conducted an experience on two different datasets, [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [AID](https://arxiv.org/abs/1608.05167) (satellite images). The first one is under the effect of gravity while the other one seems anisotropic (does not depend on direction). Those dataset are a good fit to check different network with different build in symmetries.
{: .text-justify}

## Test on CIFAR10

By looking at the result table we deduce that apart from horizontal symmetries, being invariant to other symmetries badly affect the accuracy. his might be explained by the fact that gravity plays an important role in natural images therefore you usually don't have horizontal axes symmetries. On the other hand, it is often the case that you find vertical axes symmetries in the natural world.

### Test on AID

Compare to result on CIFAR-10, we see that independently of the network architecture results are quite the same. Convolving on 2d grid graph give slightly better results than others graphs. This shows that rotation invariance for that kind of data is interesting to take into account when building the network. The reason might be again due to the nature of the data. AID images being taken from an above point of view, any rotation of the input image remains a legit input.


## Conclusion

In that blog post we showed that using the flexibility of graph convolution techniques we can test the presence of symmetries. This study highlights the fact that invariance is a property that is beneficial for the network we used appropriately, but can reduce the accuracy as well. Therefore knowing what kind of symmetry is present into your data is crucial when designing your network.

## Future work

During that experimentation the symmetries have been chosen by human (designing the underlying graph). By exploiting the flexibility of graph framework, it would be interesting to let the network learn which set of edge each underlying graph is constructed of. If we restrain the network to lean regular graph only, it should be able to learn the optimal underlying graph for each dataset, namely the one reflecting the symmetries in the dataset.  

