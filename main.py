import os
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

# from models import ConvNet, vgg11, vgg11_2, ModelC
from models.VGG import vgg11
from models.resnet import ResNet18

# TODO add it to an argparser
batch_size = 128
device = torch.device('cpu')
supported_models = ['Basic', 'Basic2', 'VGG', 'ModelC', 'ResNet18']

def get_dataloaders(dataset='CIFAR10', data_augmentation=False):
    suported_datasets = ['CIFAR10']
    if dataset not in suported_datasets:
        raise ValueError(f"Dataset {dataset} is not supported")

    if dataset == 'CIFAR10':
        base_transform = [transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])]
        augmented_transform = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4)]
        transform = transforms.Compose(base_transform)
        if data_augmentation:
            print("Augmenting data")
            transform_train = transforms.Compose(augmented_transform + base_transform)
        else:
            transform_train = transform
        X_train = torchvision.datasets.CIFAR10('dataset', train=True, transform=transform_train, download=True)
        X_test = torchvision.datasets.CIFAR10('dataset', train=False, transform=transform, download=True)
    
    dataloaders = dict()
    dataloaders['train'] = DataLoader(dataset=X_train, batch_size=batch_size, shuffle=True,
                                      num_workers=4)
    dataloaders['test'] = DataLoader(dataset=X_test, batch_size=batch_size, shuffle=False,
                                     num_workers=4)
    return dataloaders

def get_model(model_type='Basic', conv='2D'):
    if conv not in ['2D', 'graph']:
        raise ValueError(f"{conv} is not a suported type of convolution. Please use either '2D' or 'graph'")
    if model_type not in supported_models:
        raise ValueError(f"Unsuported NN architecture")
    
    if model_type == 'Basic':
        return ConvNet()
    if model_type == 'Basic2':
        return ConvNet2()
    if model_type == 'VGG':
        return vgg11()
    if model_type == 'ModelC':
        return ModelC()
    if model_type == 'ResNet18':
        return ResNet18()


def train(model, dataloader, writer, epoch, nb_epochs):
    model.train()
    losses = torch.empty(len(dataloader), requires_grad=False)
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        losses[i] = loss.item()
        writer.add_scalars('graph/loss', {'train': loss.item()}, epoch*500 + i)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, nb_epochs, i + 1, len(dataloader), loss.item(),
                          (correct / total) * 100))
            # TODO write the mean accuracy over the 100 batchs
            writer.add_scalars('graph/accuracy', {'train': (correct / total) * 100}, epoch*500 + i)
    return losses.mean().item()

def evaluate(model, dataloader, writer, epoch):
    # TODO add echo to eval
    model.eval()
    correct_sum = 0
    total_sum = 0
    test_loss = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss.append(loss.item())
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            correct_sum += correct
            total_sum += total

    writer.add_scalars('graph/loss', {'test': sum(test_loss)/ len(test_loss)}, (epoch+1)*500)
    writer.add_scalars('graph/accuracy', {'test': (correct_sum / total_sum) * 100}, (epoch+1)*500)

def adjust_learning_rate(optimizer, epoch, writer, args):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    power = 0
    if epoch > 300:
        power = 3
    if epoch > 250:
        power = 2
    if epoch > 200:
        power = 1
    lr = args.lr * (0.1 ** (power))
    writer.add_scalar('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
                        help='To set in order to apply data augmentation to the training set')
    parser.add_argument('--restore', dest='restore_from_checkpoint', action='store_true',
                        help='Restore from last checkpoint.')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    # load the model
    starting_epoch = 0
    writer = SummaryWriter(log_dir=args.log_dir)
    dataloaders = get_dataloaders('CIFAR10', data_augmentation=args.data_augmentation)

    model = get_model(args.arch)
    writer.add_graph(model, next(iter(dataloaders['train']))[0], False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        print("Model runing on CUDA")
        _ = model.cuda()
        device = torch.device('cuda')
    else:
        print("Model runing on CPU")
    
    if args.restore_from_checkpoint:
        path = os.path.join(args.checkpoints_dir, 'model.tar')
        try:
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            starting_epoch = checkpoint['epoch']
            print(f"Model restored from epoch {starting_epoch}")
        except FileNotFoundError:
            print(f"Can't restore from checkpoint as checkpoint {path} doesn't exist")


    if not os.path.isdir(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)

    # training loop
    print(f"training for {args.nb_epochs} epochs")
    for epoch in range(starting_epoch, args.nb_epochs):
        adjust_learning_rate(optimizer, epoch, writer, args)
        train(model, dataloaders['train'], writer, epoch, args.nb_epochs)
        # TODO save best model according to loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(args.checkpoints_dir, 'model.tar'))
        evaluate(model, dataloaders['test'], writer, epoch)
