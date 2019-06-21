from DG_parser import args, string_print
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim.lr_scheduler as LRS
import torch.optim as optim
from collections import deque
from DG_datasets import num_classes


class BuildDG(nn.Module):
    def __init__(self, model, optimizer, split_loc):
        super(BuildDG, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.delay = 2*(args.num_split - split_loc - 1)
        self.dg = None
        self.output = deque(maxlen=self.delay)        
        self.module_num = split_loc
        for x in range(self.delay): 
            self.output.append(None)
        self.input = deque(maxlen=self.delay+1)
        for x in range(self.delay+1):
            self.input.append(None)
        self.input_grad = None
        self.acc = 0
        self.loss = 0
        if split_loc == 0:
            self.first_layer = True
            self.last_layer = False
        elif split_loc == args.num_split-1:
            self.first_layer = False
            self.last_layer = True
        else:
            self.first_layer = False
            self.last_layer = False
        
    def forward(self, x):
        out = self.model(x)
        return out

    def backward(self):
        graph = self.output.popleft()
        if self.dg is not None and graph is not None:
            graph.backward(args.lr_shrink*self.dg)
        else:
            print('no backward in module {}'.format(self.module_num))
    
    def get_grad(self):
        return self.input.popleft().grad
    
    def get_output(self):
        return self.output[self.delay-1]
    
    def train(self):
        self.model.train()
        
    def step(self):
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()


device = {}
for i in range(args.num_split):
    device[i] = torch.device('cuda:'+str(i))
if args.backprop:
    for i in range(args.num_split):
        device[i] = torch.device('cuda:' + str(0))

model_list = {}
if args.model == 'ResNet18':
    import ResNet_ImageNet as ResNet
    model = ResNet.ResNet18(num_classes=num_classes) 
    if args.num_split == 2:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1, model.layer2)
        model_list[1] = nn.Sequential(model.layer3, model.layer4, model.avgpool, model.flatten, model.linear)
    if args.num_split == 3:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1)
        model_list[1] = nn.Sequential(model.layer2, model.layer3[:1])
        model_list[2] = nn.Sequential(model.layer3[1:], model.layer4, model.avgpool, model.flatten, model.linear)

if args.model == 'ResNet50':
    import ResNet_ImageNet as ResNet
    model = ResNet.ResNet50(num_classes=num_classes)
    if args.num_split == 2:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1, model.layer2)
        model_list[1] = nn.Sequential(model.layer3, model.layer4, model.avgpool, model.flatten, model.linear)
    if args.num_split == 3:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1, model.layer2[:2])
        model_list[1] = nn.Sequential(model.layer2[2:], model.layer3[:3])
        model_list[2] = nn.Sequential(model.layer3[3:], model.layer4, model.avgpool, model.flatten, model.linear)
    if args.num_split == 4:
        model_list[0] = nn.Sequential(model.conv1, model.bn1,model.relu, model.layer1, model.layer2[:1])
        model_list[1] = nn.Sequential(model.layer2[1:], model.layer3[:1])
        model_list[2] = nn.Sequential(model.layer3[1:5])
        model_list[3] = nn.Sequential(model.layer3[5:], model.layer4, model.avgpool, model.flatten, model.linear)


if args.model == 'ResNet101':
    import ResNet_ImageNet as ResNet
    model = ResNet.ResNet101(num_classes=num_classes)
    if args.num_split == 2:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1, model.layer2, model.layer3[:9])
        model_list[1] = nn.Sequential(model.layer3[9:], model.layer4, model.avgpool, model.flatten, model.linear)
    if args.num_split == 3:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1, model.layer2, model.layer3[:4])
        model_list[1] = nn.Sequential(model.layer3[4:15])
        model_list[2] = nn.Sequential(model.layer3[15:], model.layer4, model.avgpool, model.flatten, model.linear)
    if args.num_split == 4:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1, model.layer2, model.layer3[:1])
        model_list[1] = nn.Sequential(model.layer3[1:9])
        model_list[2] = nn.Sequential(model.layer3[9:17])
        model_list[3] = nn.Sequential(model.layer3[17:], model.layer4, model.avgpool, model.flatten, model.linear)
    if args.num_split == 5:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1, model.layer2[:3])
        model_list[1] = nn.Sequential(model.layer2[3:], model.layer3[:6])
        model_list[2] = nn.Sequential(model.layer3[6:13])
        model_list[3] = nn.Sequential(model.layer3[13:20])
        model_list[4] = nn.Sequential(model.layer3[20:], model.layer4, model.avgpool, model.flatten, model.linear)


if args.model == 'ResNet20':
    import ResNet_cifar as ResNet
    model = ResNet.resnet20(num_classes=num_classes)
    if args.num_split == 2:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1, model.layer2[:1])
        model_list[1] = nn.Sequential(model.layer2[1:], model.layer3, model.avgpool, model.flatten, model.linear)
        
    if args.num_split == 3:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1)
        model_list[1] = nn.Sequential(model.layer2)
        model_list[2] = nn.Sequential(model.layer3, model.avgpool, model.flatten, model.linear)

if args.model == 'ResNet56':
    import ResNet_cifar as ResNet
    model = ResNet.resnet56(num_classes=num_classes)
    if args.num_split == 2:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1, model.layer2[:4])
        model_list[1] = nn.Sequential(model.layer2[4:], model.layer3, model.avgpool, model.flatten, model.linear)
    if args.num_split == 3:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1)
        model_list[1] = nn.Sequential(model.layer2)
        model_list[2] = nn.Sequential(model.layer3, model.avgpool, model.flatten, model.linear)
    if args.num_split == 4:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1[:6])
        model = ResNet.resnet56(num_classes=num_classes)
        model_list[1] = nn.Sequential(model.layer1[6:], model.layer2[:4])
        model = ResNet.resnet56(num_classes=num_classes)
        model_list[2] = nn.Sequential(model.layer2[4:], model.layer3[:2])
        model = ResNet.resnet56(num_classes=num_classes)
        model_list[3] = nn.Sequential(model.layer3[2:], model.avgpool, model.flatten, model.linear)

if args.model == 'ResNet110':
    import ResNet_cifar as ResNet
    model = ResNet.resnet110(num_classes=num_classes)
    if args.num_split == 2:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1, model.layer2[:9])
        model_list[1] = nn.Sequential(model.layer2[9:], model.layer3, model.avgpool, model.flatten, model.linear)
    if args.num_split == 3:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1)
        model_list[1] = nn.Sequential(model.layer2)
        model_list[2] = nn.Sequential(model.layer3, model.avgpool, model.flatten, model.linear)
    if args.num_split == 4:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1[:13])
        model_list[1] = nn.Sequential(model.layer1[13:], model.layer2[:9])
        model_list[2] = nn.Sequential(model.layer2[9:], model.layer3[:5])
        model_list[3] = nn.Sequential(model.layer3[5:], model.avgpool, model.flatten, model.linear)
    if args.num_split == 5:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1[:10])
        model_list[1] = nn.Sequential(model.layer1[10:], model.layer2[:3])
        model_list[2] = nn.Sequential(model.layer2[3:14])
        model_list[3] = nn.Sequential(model.layer2[14:], model.layer3[:7])
        model_list[4] = nn.Sequential(model.layer3[7:], model.avgpool, model.flatten, model.linear)

if args.model == 'ResNet1202':
    import ResNet_cifar as ResNet
    model = ResNet.resnet1202(num_classes=num_classes)
    if args.num_split == 2:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1, model.layer2[:50])
        model_list[1] = nn.Sequential(model.layer2[50:], model.layer3, model.avgpool, model.flatten, model.linear)
    if args.num_split == 3:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1)
        model_list[1] = nn.Sequential(model.layer2)
        model_list[2] = nn.Sequential(model.layer3, model.avgpool, model.flatten, model.linear)
    if args.num_split == 4:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1[:75])
        model_list[1] = nn.Sequential(model.layer1[75:], model.layer2[:50])
        model_list[2] = nn.Sequential(model.layer2[50:], model.layer3[:25])
        model_list[3] = nn.Sequential(model.layer3[25:], model.avgpool, model.flatten, model.linear)
    if args.num_split == 5:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1[:60])
        model_list[1] = nn.Sequential(model.layer1[60:], model.layer2[:20])
        model_list[2] = nn.Sequential(model.layer2[20:80])
        model_list[3] = nn.Sequential(model.layer2[80:], model.layer3[:40])
        model_list[4] = nn.Sequential(model.layer3[40:], model.avgpool, model.flatten, model.linear)
        
        
if args.model == 'WRN28_10':
    import WRN as WRN
    model = WRN.wrn28_10(num_classes=num_classes)
    if args.num_split == 2:
        model_list[0] = nn.Sequential(model.conv1, model.layer1, model.layer2[:2])
        model_list[1] = nn.Sequential(model.layer2[2:], model.layer3, model.bn1, model.relu, model.avgpool, model.flatten, model.linear)
    if args.num_split == 3:
        model_list[0] = nn.Sequential(model.conv1, model.layer1)
        model_list[1] = nn.Sequential(model.layer2)
        model_list[2] = nn.Sequential(model.layer3, model.bn1, model.relu, model.avgpool, model.flatten, model.linear)
    if args.num_split == 4:
        model_list[0] = nn.Sequential(model.conv1, model.layer1[:3])
        model_list[1] = nn.Sequential(model.layer1[3:], model.layer2[:2])
        model_list[2] = nn.Sequential(model.layer2[2:], model.layer3[:1])
        model_list[3] = nn.Sequential(model.layer3[1:], model.bn1, model.relu, model.avgpool, model.flatten, model.linear)


optimizer = {}
scheduler = {}
for m in model_list:
    model_list[m] = model_list[m].to(device[m])
    if args.optim == 'adam':
        optimizer[m] = optim.Adam(model_list[m].parameters(), lr=args.lr)
    elif args.optim == 'SGD':
        optimizer[m] = optim.SGD(model_list[m].parameters(), lr=args.lr, momentum=args.momentum,
                                 weight_decay=args.weight_decay)
    else:
        print('no optimizer found!')
    scheduler[m] = LRS.MultiStepLR(optimizer[m], milestones=args.lr_decay_milestones, gamma=args.lr_decay_fact)
    scheduler[m].step()
print('Breaking {} into {} pieces.'.format(args.model, args.num_split))

ModelDg = {}
for m in range(args.num_split):
    ModelDg[m] = BuildDG(model_list[m], optimizer[m], m)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#model_cmb = Model_cmb(model_list)


'''
images = Variable(torch.randn(8,3,32,32), requires_grad=True).to(device)

model = model.to(device)

for m in model_list:
    model_list[m].train()

outputs1 = model(images)
outputs2 = images

for m in model_list:
    outputs2 = model_list[m](outputs2)
    
print(outputs1-outputs2)
'''

