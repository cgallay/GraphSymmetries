import argparse

supported_models = ['ConvNet', 'VGG', 'ResNet18']

def get_args():
    parser = argparse.ArgumentParser(
        description='Train CNN.')
    
    parser.add_argument('--arch', type=str, choices=supported_models, required=True,
                        help='model name parameter')
    parser.add_argument('--nb_epochs', type=int, default=5,
                        help='Number of epoch for the training.')
    parser.add_argument('--log_dir', type=str, default='Graph',
                        help='Path to the log directory.')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints',
                        help='Path to the checkpoint directory.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--data_augmentation', '--da', dest='data_augmentation', action='store_true',
                        help='To set in order to apply data augmentation to the training set.')
    parser.add_argument('--restore', dest='restore_from_checkpoint', action='store_true',
                        help='Restore from last checkpoint.')
    parser.add_argument('--learning_steps', '--ls', nargs='+', default=[], type=int,
                        help='Milestone when to decay learning rate by learing_gamma.')
    parser.add_argument('--learning_gamma', '--lg', type=float, default=0.1,
                        help='Learning rate mutiplicator per milestone.')
    parser.add_argument('--on_graph', dest='on_graph', action='store_true',
                        help='Use a grid graph to represent the image and perform convolutions on it.')
    parser.add_argument('--explore', dest='explore', action='store_true',
                        help='Exploration of the learning rate mode.')
    
    return parser.parse_args()