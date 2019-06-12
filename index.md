---
title:  "How to choose the image’s symmetries with graph neural network"
layout: splash
classes: wide

excerpt: "By [Charles Gallay](https://github.com/cgallay) under the supervision of [Michaël Defferrard](http://deff.ch) and [Nathanaël Perraudin](https://perraudin.info)"
header:
  overlay_image: docs/images/symmetries.png
  overlay_filter: rgba(125, 125, 125, 0.9)
  caption: "Photo credit: [**Wikipedia**](https://en.wikipedia.org/wiki/Symmetry_group)"


feature_row:
  - image_path: docs/images/2dGrid.png
    alt: "placeholder image 1"
    title: # "Placeholder 1"
    excerpt: "2dGrid graph is the simplest regular graph that is defined on the plane. Its undirected edges gives it multiple symmetries. The ones we are interested in are: rotation of 90 degrees and mirroring."

  - image_path: docs/images/vertical_horizontal.png
    alt: "placeholder image 2"
    title: # "Placeholder 2"
    excerpt: "Graphs composed of undirected vertical or horizontal edges. The symmetries found here are vertical and horizontal flips."

  - image_path: docs/images/all_direction.png
    title: # "Placeholder 3"
    excerpt: "Last type of graphs are simple directed graphs. Those graphs does not contain any symmetry but allow the information to flow through the image when combined with other graphs."

---



## Motivation:

Symmetries are naturally present in images. For instance, two similar objects appearing at different locations or being seen from different points of view are sharing symmetrical properties. These high level of symmetries induce a high degree of correlation between pixels of different images representing the same object that is worth exploiting. This blogpost provides the reader with a [framework](https://github.com/cgallay/GraphSymmetries/) that aims at testing the presence of hidden symmetries in datasets. We showed that by exploiting the knowledge of true symmetries residing in images, we can apply an appropriate strategy that does in fact improve accuracy. As frequently shown in machine learning, having some insights of your dataset has always given some advantages.
{: .text-justify}

## Exploiting Symmetries

To take advantage of those symmetries, people usually apply transformations to the input images. Newly derived images are similar enough to belong to the original class. While the information contained in those transformed images and might seem quite redundant to humans, they appear different to CNN. This allows the network to be able to extract new relevant features and  leads to better performance when applied to images close to transformed ones. Depending on the dataset, typical transformations include but are not limited to random rotations, cropping, illuminance variation and resizing. This technique, known as “Data Augmentation”, has shown to help CNN to better generalize.
{: .text-justify}


<figure style="width: 200px" class="align-right">
     <a href="docs/images/low_layer_filters.jpeg"><img src="docs/images/low_layer_filters.jpeg"></a>
    <figcaption>Figure 1: Example of filters learned by a standard CNN. Note that some filters learned are rotation of other ones.</figcaption>
</figure>

But creating artificial data introduces a lot of correlation among the weights learned during training. What if we could exploit those symmetries in a smarter way and build them directly into the network ? What if the network we design could react in a predictable way when faced with transformed data? This property is what we call equivariance. Classical CNN are translation equivariant, which allows for weight sharing.It has shown to help a lot the network to learn filters that can be shared across locations, therefore removing the need of learning the same filters for different locations. Recently [Cohen](https://github.com/tscohen/GrouPy), has demonstrated that you can design networks equivariant to any transformation from any [compact group](https://en.wikipedia.org/wiki/Compact_group). In his implementation, the weights of the filter are shared among all 90 degrees rotations of an image. As an illustration of the utility, the network would have to learn only one edge detector, while a standard CNN would have to learn two vertical and two horizontal ones (See Figure 1).
{: .text-justify}

> “Convolutional structure is not just a sufficient, but
also a necessary condition for equivariance to the
action of a compact group.” - [Risi and Shubhendu](https://arxiv.org/abs/1802.03690)

This strong relation between convolution and equivarience highlighted by Risi and Shubhendu, encouraged us to build an architecture that performs convolutions in order to be equivariant and augment their weight sharing. 


## Grid Graph symmetries

### Underlying graphs

To exploit the symmetries, we worked on graphs where convolution is invariant to permutation of neighbouring nodes. Invariance being a special case of equivariance where the output doesn’t change at all. The network we design have an invariance property to different transformations depending on the underlying graph we use. We present here, the different ones we designed and explain their group symmetries.
{: .text-justify}

{% include feature_row %}


By concatenating the outputs of the convolution applied on those different underlying graphs, we can build networks that are invariant to the desired symmetries only. For example, in the case of the graph below, we have a GCNN that is only invariant to horizontal mirroring. (TODO add the graph)
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
     <tr> <td>None (Directed)</td><td>66.6%</td><td>68.5%</td> </tr>
</tbody></table>
<figcaption>Table 1: 
On CIFAR-10 training was performed with 20% of the dataset and models of roughly 52’000 parameters. While on AID training was performed with 20% of the dataset downsized to a shape of 200x200 pixels and models of roughly 107’000 parameters.</figcaption>
</figure>

### Datasets
Coming from all that background, we want to check which symmetries is worth exploiting. We conducted an experience on two different datasets for a classification task, [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [AID](https://arxiv.org/abs/1608.05167) (satellite images). The first one is under the effect of gravity while the other one seems anisotropic (does not depend on direction). Those datasets are a good fit to check invariance build into networks for different types of symmetries as shown below.
{: .text-justify}

### Test on CIFAR10

By looking at the result table, we deduce that apart from horizontal symmetries being invariant to other ones, badly affects the accuracy. It might be explained by the fact that gravity plays an important role in natural images, therefore you usually don't have vertical symmetries. On the other hand, it is often the case that you find horizontal symmetries in the natural world.
{: .text-justify}

### Test on AID

Compared to CIFAR-10, we see that, independently of the network architecture, results are quite the same. Convolving on 2d grid graph gives slightly better results than others graphs. This shows that rotation invariance for that kind of data is interesting to take into account when building the network. The reason might be again due to the nature of the data. As AID images are taken from an above point of view, any rotation of the input image remains a legit input.
{: .text-justify}

## Conclusion

In that blog post we showed that by using the flexibility of graph convolution techniques, we can test the presence of symmetries. This study highlights the fact that invariance is a property that is beneficial for the network if used appropriately, but can reduce the accuracy as well. Therefore knowing what kind of symmetry is present into your data is crucial when designing your network.
{: .text-justify}


## Future work

During that experimentation the invariance to symmetries have been chosen by human (designing the underlying graph). By exploiting the flexibility of graph framework, it would be interesting to let the network learn the underlying graphs. If we restrain the network to learn regular graph only, it should be able to learn the optimal ones for each dataset, namely the ones reflecting the symmetries hidden in the dataset. 
{: .text-justify}

