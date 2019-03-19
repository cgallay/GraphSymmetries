import os
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt

from models.VGG import vgg11
from models.resnet import ResNet18
from models.classics import ConvNet
from utils.transforms import ToGraph
from utils.helpers import get_number_of_parma
from utils.argparser import get_args

# TODO add it to an argparser
batch_size = 128
device = torch.device('cpu')


def get_dataloaders(dataset='CIFAR10', data_augmentation=False, on_graph=False):
    suported_datasets = ['CIFAR10']
    if dataset not in suported_datasets:
        raise ValueError(f"Dataset {dataset} is not supported")

    if dataset == 'CIFAR10':
        train_trans = []
        test_trans = []
        if data_augmentation:
            print("Augmenting data")
            train_trans = [transforms.RandomHorizontalFlip(), 
                               transforms.RandomCrop(32, 4)]

        base_trans = [transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465),
                                           (0.2023, 0.1994, 0.2010))]
        train_trans += base_trans
        test_trans += base_trans
        
        if on_graph:
            train_trans.append(ToGraph())
            test_trans.append(ToGraph())
        
        X_train = torchvision.datasets.CIFAR10('dataset', train=True,
                    transform=transforms.Compose(train_trans), download=True)
        X_test = torchvision.datasets.CIFAR10('dataset', train=False,
                    transform=transforms.Compose(test_trans), download=True)
    
    dataloaders = dict()
    dataloaders['train'] = DataLoader(dataset=X_train, batch_size=batch_size, shuffle=True,
                                      num_workers=4)
    dataloaders['test'] = DataLoader(dataset=X_test, batch_size=batch_size, shuffle=False,
                                     num_workers=4)
    return dataloaders

def get_model(model_type='Basic', on_graph=False, device='cpu'):
    
    if model_type == 'ConvNet':
        return ConvNet(on_graph=on_graph, device=device)
    if model_type == 'VGG':
        return vgg11()
    if model_type == 'ResNet18':
        return ResNet18()

    raise ValueError(f"Unsuported NN architecture")


def train_eval(model, dataloaders, optimizer, train=True):
    if train:
        model.train()
    else:
        model.eval()
    dataloader = dataloaders['train' if train else 'test']
    criterion = nn.CrossEntropyLoss()
    losses = 0
    accurcies = 0
    for i, (images, labels) in enumerate(dataloader):
        with torch.set_grad_enabled(train):
            images = images.to(device)
            labels = labels.to(device)
            nb_item = labels.shape[0]
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses += loss.item()
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            accurcy = float((predicted == labels).sum()) / nb_item
            accurcies += accurcy
            if i % 10 == 0:
                print('Step [{}/{}],\tLoss: {:.4f},\tAccuracy: {:.2f}%\t'
                    .format(i + 1, len(dataloader), loss, accurcy * 100.0),
                    end='\r')

    metrics = {
        'loss': losses / len(dataloader),
        'accuracy': accurcies / len(dataloader)
    }
    return metrics


if __name__ == '__main__':
    global args
    args = get_args()

    # load the model
    starting_epoch = 0
    writer = SummaryWriter(log_dir=args.log_dir)
    dataloaders = get_dataloaders('CIFAR10', data_augmentation=args.data_augmentation,
                                  on_graph=args.on_graph)
    model = None
    if torch.cuda.is_available():
        print("Model runing on CUDA")
        device = torch.device('cuda')
        model = get_model(args.arch, args.on_graph, device=device)
        _ = model.cuda()
    else:
        print("Model runing on CPU")
        model = get_model(args.arch, args.on_graph, device=device)

    print(f"The model has {get_number_of_parma(model)} parameters to learn.")

    try:
        writer.add_graph(model, next(iter(dataloaders['train']))[0], False)
    except:
        print("Graph could not be loged in tensordboard.")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay)

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

    if args.explore:
        scheduler = ExponentialLR(optimizer, 2.0)
    else:
        scheduler = MultiStepLR(optimizer, milestones=args.learning_steps, gamma=args.learning_gamma,
                            last_epoch=starting_epoch-1)
    # training loop
    print(f"training for {args.nb_epochs} epochs")
    losses = []
    learning_rates = []
    for epoch in range(starting_epoch, args.nb_epochs):
        scheduler.step()
        writer.add_scalar('learning_rate/lr', scheduler.get_lr()[0], epoch)
        for step in ['train', 'test']:
            metrics = train_eval(model, dataloaders, optimizer, step == 'train')
            if step == 'test':
                losses.append(metrics['loss'])
            print('{}\tEpoch [{}/{}],\tLoss: {:.4f},\tAccuracy: {:.2f}%\t'
                  .format(step, epoch, args.nb_epochs, metrics['loss'], metrics['accuracy'] * 100))
       
        # learning_rates.append(scheduler.get_lr()[0])
        # TODO save best model according to loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(args.checkpoints_dir, 'model.tar'))
        # evaluate(model, dataloaders['test'], writer, epoch)
        # if args.explore:
        #    fig=plt.figure()
        #    ax = fig.add_subplot(1,1,1)
        #    ax.clear()
        #    ax.set_ylabel('Losses')
        #    ax.set_xlabel('learning rate')
        #    ax.plot(learning_rates, losses)
        #    writer.add_figure('learing_rate', fig)
