import argparse
from utils.config import conf

supported_models = ['ConvNet', 'VGG', 'ResNet18']
args = None


def get_args():
    global args
    if args:
        return args
    parser = argparse.ArgumentParser(
        description='Train CNN.')
    
    parser.add_argument('--arch', type=str, choices=supported_models, required=True,
                        help='model name parameter')
    parser.add_argument('--nb_epochs', type=int, default=5,
                        help='Number of epoch for the training.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for the training')
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
    parser.add_argument('--opti', type=str, choices=['Adam', 'SGD'], default='SGD',
                        help='Choice of the optimizer.')
    parser.add_argument('--L_scale', type=float, default=1.0,
                        help='scale the eingen values of the laplacian into [-L_scale, L_scale]')
    parser.add_argument('--diagonals', dest='diagonals', action='store_true',
                        help='Augment the grid graph with diagonals when set, where the weight of the edges is 1/srt(2)')
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'AID'], default='CIFAR10',
                        help='Dataset on which to run the experiment.')
    parser.add_argument('--global_average_pooling', '--GAP', dest='global_average_pooling', action='store_true',
                        help='Apply global averge pooling as last layer instead of a fully connected in order to enforce invariance')
    parser.add_argument('--vertical_graph', 'V_Graph', dest='vertical_graph', action='store_true',
                        help='Perform the convolution on two different graphs that are composed of vertical and horizontal edges only')


    parser.set_defaults(**conf)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning some unkown parameters have been defined: {unknown}")
    return args
