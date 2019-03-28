import torch

conf = {
    'fully_connected': True,  # When set to False a GAP (Global Average Pooling) layer is used.
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'

}