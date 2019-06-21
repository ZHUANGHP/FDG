import numpy as np
import torch
from torchvision import datasets, transforms
from DG_parser import args


kwargs = {'num_workers': 0, 'pin_memory': True}
if args.dataset == 'CIFAR10':
    input_dim = 32
    input_ch = 3
    num_classes = 10

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset_train = datasets.CIFAR10(root='../../data/CIFAR10/', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../../data/CIFAR10/', train=False,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)

elif args.dataset == 'CIFAR100':
    input_dim = 32
    input_ch = 3
    num_classes = 100
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])
    dataset_train = datasets.CIFAR100('../../data/CIFAR100', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../../data/CIFAR100', train=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
                          ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)