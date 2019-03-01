import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

from models import ConvNet

# TODO add it to an argparser
batch_size = 100
nb_epochs = 5
device = torch.device('cpu')
CHECKPOINT_FOLDER = 'checkpoints/'
restore_from_checkpoint = True

def get_dataloaders(dataset='CIFAR10', transform=transforms.Compose([transforms.ToTensor()])):
    suported_datasets = ['CIFAR10']
    if dataset not in suported_datasets:
        raise ValueError(f"Dataset {dataset} is not supported")

    if dataset == 'CIFAR10':
        X_train = torchvision.datasets.CIFAR10('dataset', train=True, transform=transform, download=True)
        X_test = torchvision.datasets.CIFAR10('dataset', train=False, transform=transform, download=True)
    
    dataloaders = dict()
    dataloaders['train'] = DataLoader(dataset=X_train, batch_size=batch_size, shuffle=True)
    dataloaders['test'] = DataLoader(dataset=X_test, batch_size=batch_size, shuffle=True)
    return dataloaders

def get_model(model_type='Basic', conv='2D'):
    if conv not in ['2D', 'graph']:
        raise ValueError(f"{conv} is not a suported type of convolution. Please use either '2D' or 'graph'")
    supported_models = ['Basic']
    if model_type not in supported_models:
        raise ValueError(f"Unsuported NN architecture")
    
    if model_type == 'Basic':
        return ConvNet()

def train(model, dataloader, writer):
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
            writer.add_scalars('graph/accuracy', {'train': (correct / total) * 100}, epoch*500 + i)
    return losses.mean().item()

def evaluate(model, dataloader, writer):
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


if __name__ == '__main__':
    # load the model
    starting_epoch = 0
    dataloaders = get_dataloaders('CIFAR10')

    model = get_model('Basic')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if torch.cuda.is_available():
        print("Model runing on CUDA")
        _ = model.cuda()
        device = torch.device('cuda')
    else:
        print("Model runing on CPU")
    
    
    if restore_from_checkpoint:
        path = os.path.join(CHECKPOINT_FOLDER, 'model.tar')
        try:
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            starting_epoch = checkpoint['epoch']
            print(f"Model restored from epoch {starting_epoch}")
        except FileNotFoundError:
            print(f"Can't restore from checkpoint as checkpoint {path} doesn't exist")

    
    writer = SummaryWriter(log_dir='Graph')

    if not os.path.isdir(CHECKPOINT_FOLDER):
        os.makedirs(CHECKPOINT_FOLDER)

    # training loop
    for epoch in range(starting_epoch, nb_epochs):
        train(model, dataloaders['train'], writer)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(CHECKPOINT_FOLDER, 'model.tar'))
        evaluate(model, dataloaders['test'], writer)
