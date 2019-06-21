from DG_parser import args, string_print
import torch
import torch.nn as nn

import torch.multiprocessing as mp
import threading as thread
import time
from torch.autograd import Variable
import torch.nn.functional as F
from DG_datasets import train_loader, test_loader
from tensorboardX import SummaryWriter
from DG_models import ModelDg, scheduler, device
import queue

# import torch.backends.cudnn as cudnn
# cudnn.benchmark = True

# args.writer = True
if args.writer:
    writer = SummaryWriter()


class Info:
    def __init__(self, num):
        self.train_error = None
        self.train_loss = None
        self.test_error = None
        self.test_loss = None
        self.string = None
        self.lr = torch.zeros(num)
        self.epoch = None
        self.time = None


info = Info(args.num_split)


def TrainDg(model, images, labels):
    name = thread.currentThread().getName()
    if not model.last_layer:
        if images is not None:
            model.train()
            model.zero_grad()
            model.backward()
            model.step()
            outputs = model(images)
            model.output.append(outputs)
            if not model.first_layer:
                model.input.append(images)
                input_images = model.input.popleft()
                if input_images is None:
                    print('no input gradients obtained in module {}'.format(model.module_num))
                model.input_grad = input_images.grad if input_images is not None else None
            # communication in thread
            model.zero_grad()
    #            print('arriving data..')
    elif model.last_layer:
        if images is not None and labels is not None:
            model.train()
            model.step()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc = correct / images.size(0)
            model.zero_grad()
            loss.backward()
            model.input_grad = images.grad
            model.acc = acc
            model.loss = loss.item()


def train_bp(model, images, labels):
    outputs = images
    for m in range(args.num_split):
        model[m].model.train()
        outputs = model[m](outputs)

    loss = F.cross_entropy(outputs, labels)
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    acc = correct / images.size(0)
    for m in range(args.num_split):
        model[m].zero_grad()

    loss.backward()
    for m in range(args.num_split):
        model[m].step()

    return acc, loss.item()


def test(model):
    for m in range(args.num_split):
        model[m].model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        loss_tm = 0
        for images, labels in test_loader:
            outputs = images
            labels = labels.to(device[args.num_split - 1])
            for m in model:
                outputs = outputs.to(device[m])
                outputs = model[m](outputs)
            loss = F.cross_entropy(outputs, labels)
            loss_tm += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        info.test_error = 1 - correct / total
        info.test_loss = loss_tm / len(test_loader)
    print(
        '{}, Epochs: {}/{}, lr: {:.5f}, Train Loss: {:.4f}, Val Loss: {:.4f}, Train error: {:.5f}, Test err: {:.5f}, used time: {:.5f}'.format(
            string_print,
            info.epoch + 1,
            args.epochs,
            info.lr[0],
            info.train_loss,
            info.test_loss,
            info.train_error,
            info.test_error,
            info.time))

    if args.writer:
        writer.add_scalar(string_print + '/err', info.test_error, epoch + 1)
        writer.add_scalar(string_print + '/trainerr', info.train_error, epoch + 1)
        writer.add_scalar(string_print + '/lr', info.lr[0], epoch + 1)

        writer.add_scalar(string_print + '/loss_train', info.train_loss, epoch + 1)
        writer.add_scalar(string_print + '/loss_test', info.test_loss, epoch + 1)


# info.lr = args.lr

if args.warm_up:
    # warming up for 3 epochs and reset to the original lr
    for m in ModelDg:
        for param_group in ModelDg[m].optimizer.param_groups:
            param_group['lr'] = 0.1 * args.lr

iters = 0
end_time = time.time()
epoch_start = 0
input_info = {}
target_info = {}
last_idx = args.num_split - 1
labels_q = queue.Queue()
for i in range(args.num_split):
    input_info[i], target_info[i] = None, None

for i in range(args.num_split - 2):
    labels_q.put(None)

input_app = []

print('Training begins')
if args.warm_up:
    print('warming up...')
for epoch in range(epoch_start, args.epochs):
    info.epoch = epoch
    if args.warm_up:
        # warming up for 3 epochs and reset to the original lr
        if epoch == 3:
            print('warm-up completed.')
            for m in ModelDg:
                for param_group in ModelDg[m].optimizer.param_groups:
                    param_group['lr'] = args.lr
    train_loss_sum = 0
    train_loss_tmp = 0
    acc_sum = 0
    for i, (images, labels) in enumerate(train_loader):
        iters += 1
        images = Variable(images.to(device[0], non_blocking=True), requires_grad=True)
        labels = Variable(labels.to(device[0], non_blocking=True), requires_grad=False)
        # Move tensors to the configured device
        # newest batch input
        input_info[0] = images
        # input_app.append(images)
        labels_q.put(labels)

        if args.backprop:
            labels = labels.to(device[0])
            acc, train_loss_tmp = train_bp(ModelDg, images.to(device[0]), labels.to(device[args.num_split - 1]))
            acc_sum += acc
            train_loss_sum += train_loss_tmp
        else:
            processes = []

            '''
            for m in range(args.num_split):
                p = mp.Process(target=TrainDg, args=(ModelDg[m], input_info[m], target_info[m]))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            '''

            # porblems could happen if run in threads
            for m in range(args.num_split):
                p = thread.Thread(name=str(m), target=TrainDg, args=(ModelDg[m], input_info[m], target_info[m]))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

            '''
            for i in range(args.num_split):
                TrainDg(ModelDg[i], input_info[i], target_info[i])
            '''
            # passing inputs
            for i in reversed(range(1, args.num_split)):
                tmp = ModelDg[i - 1].get_output()
                input_info[i] = Variable(tmp.detach().clone().to(device[i], non_blocking=True),
                                         requires_grad=True) if tmp is not None else None

            for i in range(args.num_split - 1):
                ModelDg[i].dg = ModelDg[i + 1].input_grad.clone().to(device[i], non_blocking=True) if ModelDg[
                                                                                       i + 1].input_grad is not None else None
            labels_last = labels_q.get()
            target_info[last_idx] = labels_last.to(device[last_idx], non_blocking=True) if labels_last is not None else None
            acc_sum += ModelDg[last_idx].acc
            train_loss_sum += ModelDg[args.num_split - 1].loss
    # summarizing the info for each epoch to print
    info.train_error = 1 - acc_sum / len(train_loader)
    info.train_loss = train_loss_sum / len(train_loader)
    info.time = time.time() - end_time
    end_time = time.time()
    # update lr and get lr
    for m in range(args.num_split):
        scheduler[m].step(epoch)
        lr = scheduler[m].get_lr()
        info.lr[m] = lr[0]
    # test the network
    test(ModelDg)
writer.close()



