from utils.helpers import conv_output_shape
from torch.nn import MaxPool2d, Conv2d


def get_conv(**kwargs):    
    # accept only args of the function __init__
    f_kwargs = {k:v for k,v in kwargs.items() if k in Conv2d.__init__.__code__.co_varnames}
    conv = Conv2d(**f_kwargs)

    f_kwargs = {k:v for k,v in kwargs.items() if k in conv_output_shape.__code__.co_varnames}
    f_kwargs['h_w'] = kwargs['input_shape']
    out_shape = conv_output_shape(**f_kwargs)
    
    return conv, out_shape


def get_pool(**kwargs):
    f_kwargs = {k:v for k,v in kwargs.items() if k in MaxPool2d.__init__.__code__.co_varnames}
    pool = MaxPool2d(**f_kwargs)

    f_kwargs = {k:v for k,v in kwargs.items() if k in conv_output_shape.__code__.co_varnames}
    f_kwargs['h_w'] = kwargs['input_shape']
    out_shape = conv_output_shape(**f_kwargs)
    return pool, out_shape

