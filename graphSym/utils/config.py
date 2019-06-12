import torch

conf = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'

}
