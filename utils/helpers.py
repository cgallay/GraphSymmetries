from functools import reduce

def get_number_of_parma(model):
    """
    Get the number of trainable parameter of a pytorch model
    """
    nb_param = 0
    for param in model.parameters():
        nb_param+=reduce(lambda x, y: x*y, param.shape)
    return nb_param