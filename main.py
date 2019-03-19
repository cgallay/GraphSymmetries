import os
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter

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


def train(model, dataloader, writer, epoch, nb_epochs):
    model.train()
    losses = torch.empty(len(dataloader), requires_grad=False)
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        losses[i] = loss.item()
        writer.add_scalars('graph/loss', {'train': loss.item()}, epoch*len(dataloaders['train']) + i)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        writer.add_scalars('graph/accuracy', {'train': (correct / total) * 100}, epoch*len(dataloaders['train']) + i)
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, nb_epochs, i + 1, len(dataloader), loss.item(),
                          (correct / total) * 100))
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

    writer.add_scalars('graph/loss', {'test': sum(test_loss)/ len(test_loss)}, (epoch+1)*len(dataloaders['train']))
    writer.add_scalars('graph/accuracy', {'test': (correct_sum / total_sum) * 100}, (epoch+1)*len(dataloaders['train']))

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

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay)

    if args.restore_from_checkpoint:
        path = os.path.join(args.checkpoints_dir, 'model.tar')
        try:
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            starting_epoch = checkpoint['epoch']
            # TODO restore learning_steps
            print(f"Model restored from epoch {starting_epoch}")
        except FileNotFoundError:
            print(f"Can't restore from checkpoint as checkpoint {path} doesn't exist")


    if not os.path.isdir(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)

    scheduler = MultiStepLR(optimizer, milestones=args.learning_steps, gamma=args.learning_gamma,
                            last_epoch=starting_epoch-1)
    # training loop
    print(f"Training for {args.nb_epochs} epochs")
    for epoch in range(starting_epoch, args.nb_epochs):
        scheduler.step()
        writer.add_scalar('learning_rate', scheduler.get_lr()[0], epoch)
        train(model, dataloaders['train'], writer, epoch, args.nb_epochs)
        # TODO save best model according to loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(args.checkpoints_dir, 'model.tar'))
        evaluate(model, dataloaders['test'], writer, epoch)
