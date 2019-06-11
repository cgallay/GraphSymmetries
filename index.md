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
    excerpt: "2dGrid graph is the simplest regular graph that is defined on the plane. It's undirected edges gives him multiple symmetries. The ones we are interested in are: rotation of 90 degrees and mirroring."

  - image_path: docs/images/vertical_horizontal.png
    alt: "placeholder image 2"
    title: # "Placeholder 2"
    excerpt: "Graphs composed of undirected vertical or horizontal edges. The symmetries found here are vertical and horizontal flips."

  - image_path: docs/images/all_direction.png
    title: # "Placeholder 3"
    excerpt: "Last type of graphs are simple directed graph. Those graph does not contain any symmetry but allow the information to flow across the image when combined with other graphs."

---



## Motivation:

Symmetries are naturally present in images. From two similar objects appearing in different locations, to different points of view. This high level of symmetries induce a high degree of correlation between pixels of different image representing the same object that is worth exploiting. This blogpost provide the reader with a [framework](https://github.com/cgallay/GraphSymmetries/) that aim at testing symmetries hidden in datasets. we showed that exploiting the true symmetry does infact boost performances. Once again showing that knowing your dataset is something that still has a huge impact.
{: .text-justify}


## Exploiting Symmetries

To exploit those symmetries people usually apply transformations to the input image. Newly synthetic images have the property of still belonging to the original class. While those images might seem obviously the same for humans, for CNN they appear as quite different ones. This allows the network to learn news features extractor and perform better on images close to the transformed one. Depending on the dataset typical transformation include but are not limited to  random rotations, cropping, illuminance variation and resizing. This technique called “Data Augmentation” has been shown to help CNN to better generalize.
{: .text-justify}


<figure style="width: 200px" class="align-right">
     <a href="docs/images/low_layer_filters.jpeg"><img src="docs/images/low_layer_filters.jpeg"></a>
    <figcaption>Figure 1: Example of filters learned by a standard CNN. Note that some filters learn are rotation of other ones.</figcaption>
</figure>

But creating those artificial data introduce a lot of correlation into the wights learned during training. What if you could exploit those symmetry in a smarter way and build them directly into the network ? What if the network you design could react in a predictable way when faced with transformed data? This property is what we call equivariance. In the case of classical CNN, we have an equivariance to translation which allows weight sharing. It has been shown to help a lot the network to learn kernel that can be shared across locations. Recently [Cohen](https://github.com/tscohen/GrouPy), has shown that you can design network equivariant to any transformation from any compact group. In his implementation the weights of the filter are shared across 90 degree rotation of the input. As an illustration of the utility, the network will have to learn only one edge detector, while a standard CNN would have to learn two vertical and two horizontal ones (See Figure 1).
{: .text-justify}

> “Convolutional structure is not just a sufficient, but
also a necessary condition for equivariance to the
action of a compact group.” - [Risi and Shubhendu](https://arxiv.org/abs/1802.03690)

This strong relation between convolution and equivarience highlight by Risi and Shubhendu, encouraged us to build architecture that perform convolution in order to be equivariant and to do even more weight sharing. 


## Grid Graph symmetries

### Underlying graphs

To exploit the symmetries, we works on graph where convolution is invariant to permutation of neighbouring nodes. Invariance being a special case of equivariance where the output doesn’t change at all. Network we design will have invariance property to different transformation depending on the underlying graph we use. We present here the different underlying graph we used and explain their group symmetry.
{: .text-justify}

{% include feature_row %}


By concatenation the output of the Graph Convolution on those different underlying graph, we can build network that are invariant to desire symmetries only. For example in the case of the graph below, we have a GCNN that is only invariant to horizontal mirroring.
{: .text-justify}


## Experimentation

<figure class="align-right" style="width: 325px">
<table class="sortable">
 <caption>Validation Accuracy</caption>
<thead> <tr> <th>Symmeties</th> <th>CIFAR-10</th> <th>AID</th> </tr></thead>
<tbody>
     <tr> <td>All (2dGrid)</td><td>52.9%</td> <td>70.4%</td> </tr>
     <tr> <td>Vertical</td><td>57.8%</td><td>-</td> </tr>
     <tr> <td>Vertical+Horizontal</td><td>60.5%</td> <td>67.5%</td> </tr>
     <tr> <td>Horizontal</td><td>64.3%</td><td>-</td> </tr>
     <tr> <td>None (Directed)</td><td>66.6%</td><td>xx.x%</td> </tr>
</tbody></table>
<figcaption>Table 1: 
On CIFAR-10 training was performed with 20% of the dataset and models of roughly 52’000 parameters. While on AID training was performed with 20% of the dataset downsized to a shape of 200x200 pixels and models of roughly 107’000 parameters.</figcaption>
</figure>

### Dataset
Coming from all that background, we want to check which symeties is worth exploiting. We conducted an experience on two different datasets for a classification task, [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [AID](https://arxiv.org/abs/1608.05167) (satellite images). The first one is under the effect of gravity while the other one seems anisotropic (does not depend on direction). Those dataset are a good fit to check different network with different build in symmetries.
{: .text-justify}

### Test on CIFAR10

By looking at the result table we deduce that apart from horizontal symmetries, being invariant to other symmetries badly affect the accuracy. his might be explained by the fact that gravity plays an important role in natural images therefore you usually don't have horizontal axes symmetries. On the other hand, it is often the case that you find vertical axes symmetries in the natural world.
{: .text-justify}

### Test on AID

Compare to result on CIFAR-10, we see that independently of the network architecture results are quite the same. Convolving on 2d grid graph give slightly better results than others graphs. This shows that rotation invariance for that kind of data is interesting to take into account when building the network. The reason might be again due to the nature of the data. AID images being taken from an above point of view, any rotation of the input image remains a legit input.
{: .text-justify}

## Conclusion

In that blog post we showed that using the flexibility of graph convolution techniques we can test the presence of symmetries. This study highlights the fact that invariance is a property that is beneficial for the network if used appropriately, but can reduce the accuracy as well. Therefore knowing what kind of symmetry is present into your data is crucial when designing your network.
{: .text-justify}


## Future work

During that experimentation the invariance to symmetries have been chosen by human (designing the underlying graph). By exploiting the flexibility of graph framework, it would be interesting to let the network learn the underlying graphs. If we restrain the network to learn regular graph only, it should be able to learn the optimal ones for each dataset, namely the ones reflecting the symmetries hidden in the dataset. 
{: .text-justify}

