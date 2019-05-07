import os
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR

from models.graph_aid import GraphConvNetAID
from models.cnn_aid import CNNConvNetAID
from models.graph_cifar import GraphConvNetCIFAR
from models.cnn_cifar import CNNConvNetCIFAR

from utils.transforms import ToGraph
from utils.helpers import get_number_of_parma
from utils.argparser import get_args, repr_args
from utils.logger import Logger
args = get_args()

# TODO add it to an argparser
batch_size = args.batch_size
device = torch.device('cpu')
criterion = nn.CrossEntropyLoss()

def get_dataloaders(dataset='CIFAR10', data_augmentation=False, on_graph=False):
    suported_datasets = ['CIFAR10', 'AID']
    if dataset not in suported_datasets:
        raise ValueError(f"Dataset {dataset} is not supported")

    dataloaders = {}

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
        perc = 0.2
        t_size = len(X_train)
        t_sub_size = int(perc * t_size)
        X_train, _ = torch.utils.data.random_split(X_train, [t_sub_size, t_size - t_sub_size])
        print(f"Working with {len(X_train)} images which is {perc} of the dataset") 

        X_test = torchvision.datasets.CIFAR10('dataset', train=False,
                    transform=transforms.Compose(test_trans), download=True)
    
    if dataset == 'AID':
        if on_graph:
            trans = [transforms.ToTensor(), ToGraph()]
        else:
            trans = [transforms.ToTensor()]
        dataset = ImageFolder('dataset/AID/', transform=transforms.Compose(trans))
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        torch.manual_seed(42)
        X_train, X_test = random_split(dataset, [train_size, test_size])


    dataloaders['train'] = DataLoader(dataset=X_train, batch_size=batch_size, shuffle=True,
                                      num_workers=4)
    dataloaders['test'] = DataLoader(dataset=X_test, batch_size=batch_size, shuffle=False,
                                     num_workers=4)
    return dataloaders

def get_model(model_type='ConvNet', on_graph=False, device='cpu'):
    
    if (model_type, args.dataset, args.on_graph) ==\
        ('ConvNet', 'AID', True):
        return GraphConvNetAID()
    if (model_type, args.dataset, args.on_graph) ==\
        ('ConvNet', 'AID', False):
        return CNNConvNetAID()
    if (model_type, args.dataset, args.on_graph) ==\
            ('ConvNet', 'CIFAR10', True):
        return GraphConvNetCIFAR()
    
    if (model_type, args.dataset, args.on_graph) ==\
            ('ConvNet', 'CIFAR10', False):
        return CNNConvNetCIFAR()




    raise ValueError(f"Unsuported NN architecture")


def train_eval(model, dataloaders, optimizer, train=True):
    if train:
        model.train()
    else:
        model.eval()
    dataloader = dataloaders['train' if train else 'test']
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
            accurcy = float((predicted == labels).sum().item()) / nb_item
            accurcies += accurcy
            if train:
                logger.write({'loss': loss.item(), 'accuracy': accurcy})
            if i % 10 == 0:
                print("                                                                  ", end='\r')
                print('Step [{}/{}],\tLoss: {:.4f},\tAccuracy: {:.2f}%\t'
                    .format(i + 1, len(dataloader), loss, accurcy * 100.0),
                    end='\r', flush=True)

    metrics = {
        'loss': losses / len(dataloader),
        'accuracy': accurcies / len(dataloader)
    }
    return metrics


if __name__ == '__main__':
    global logger
    args = get_args()
    logger = Logger(f'Graph/{repr_args(args)}')
    logger.write_hparam(args.__dict__)
    # load the model
    starting_epoch = 0
    
    dataloaders = get_dataloaders(args.dataset, data_augmentation=args.data_augmentation,
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

    logger.write_graph(model, next(iter(dataloaders['train']))[0])

    if args.opti == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=0.1,
                                     weight_decay=args.weight_decay)
    else:
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
        logger.write({'learning rate': scheduler.get_lr()[0]}, index=epoch)

        for step in ['train', 'test']:
            metrics = train_eval(model, dataloaders, optimizer, step == 'train')
            logger.write(metrics, curve=f"mean_{step}", increment=False)
            print("                                                                  ", end='\r')
            print('{}\tEpoch [{}/{}],\tLoss: {:.4f},\tAccuracy: {:.2f}%\t'
                  .format(step, epoch, args.nb_epochs, metrics['loss'], metrics['accuracy'] * 100),
                  flush=True)

        learning_rates.append(scheduler.get_lr()[0])
        # TODO save best model according to loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(args.checkpoints_dir, 'model.tar'))
        if args.explore:
            logger.plot_eploration(learning_rates, losses)
