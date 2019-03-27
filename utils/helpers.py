from functools import reduce

def get_number_of_parma(model):
    """
    Get the number of trainable parameter of a pytorch model
    """
    nb_param = 0
    for param in model.parameters():
        nb_param+=reduce(lambda x, y: x*y, param.shape)
    return nb_param

def t_add(t, v):
    """
    Add value v to each element of the tuple t.
    """
    return tuple(i + v for i in t)


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w