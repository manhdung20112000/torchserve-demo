import torch
import torchvision
import torchvision.transforms as transforms

DATA_PATH = './data'

def get_transform():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    return transform

def get_dataset(transform, download=True):
    train_set = torchvision.datasets.MNIST(root=DATA_PATH, train=True, download=download, transform=transform)
    test_set  = torchvision.datasets.MNIST(root=DATA_PATH, train=False, download=download, transform=transform)
    return train_set, test_set

def get_dataloader(train_set, test_set, bs):
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False)

    return trainloader, testloader
