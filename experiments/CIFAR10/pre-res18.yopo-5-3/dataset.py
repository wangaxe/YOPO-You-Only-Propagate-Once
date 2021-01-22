import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

# cifar10_mean = (0.4914, 0.4822, 0.4465)
# cifar10_std = (0.2471, 0.2435, 0.2616)

def create_train_dataset(batch_size = 128, root = '/mnt/storage0_8/torch_datasets/cifar-data'):

    transform_train = transforms.Compose([
     transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    #trainset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

    return trainloader
def create_test_dataset(batch_size = 128, root = '/mnt/storage0_8/torch_datasets/cifar-data'):
    transform_test = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    #testset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, pin_memory=True, num_workers=2)
    return testloader


if __name__ == '__main__':
    print(create_train_dataset())
    print(create_test_dataset())

