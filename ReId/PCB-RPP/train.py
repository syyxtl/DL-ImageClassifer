import numpy
import matplotlib 
import torch
import torch.nn as nn
import torch.optim as optim

import os
import time
import argparse
from model import PCB
from utils import load_data

from shutil import copyfile
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--save_dir', default='./Models/', type=str, help='save model dir')
parser.add_argument('--RPP', default=True, action='store_true', help='use RPP')
args = parser.parse_args()

SAVE_DIR = args.save_dir
RPP = args.RPP

######################################################################
#************************* GPU concern  *****************************#
use_gpu = torch.cuda.is_available()
str_ids = args.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

is_parallel_train = False
if len(gpu_ids) > 1:
    is_parallel_train = True

seed = 1994
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
#************************* GPU concern  *****************************#
######################################################################

datas = load_data()

######################################################################
#************************* Save model  ******************************#
def save_network(network, epoch_label, stage):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join(SAVE_DIR, stage)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    if is_parallel_train:
        torch.save(network.module.state_dict(), save_path + '/' + save_filename)
    else:
        torch.save(network.cpu().state_dict(), save_path + '/' + save_filename)
        if torch.cuda.is_available():
            network.cuda(gpu_ids[0])
#************************* Saved model  *****************************#
######################################################################

######################################################################
#************************* load model *******************************#
def load_network(network, stage):
    save_path = os.path.join(SAVE_DIR, stage, 'net_last.pth')
    network.load_state_dict(torch.load(save_path))
    return network
#************************* loaded model *****************************#
######################################################################

######################################################################
#***************************** train model **************************#
gpu = torch.cuda.is_available()
NumParts = 6
def _train(model, criterion, optimizer, scheduler, log_file, stage, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    last_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 15)

        for phase in ["train"]:
            if phase == "train":        
                model.train(True)
            else:
                model.train(False)

            all_loss = 0.0
            all_corrects = 0.0
            for data in datas[phase]:
                inputs, labels = data

                if gpu == True:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                part = {}
                results = nn.Softmax(dim=1)

                for i in range(NumParts):
                    part[i] = outputs[i]

                score = results(part[0]) + results(part[1]) + results(part[2]) + results(part[3]) + results(part[4]) + results(part[5])
                _, preds = torch.max(score.data, 1)

                loss = 0
                for i in range(NumParts):
                    loss += criterion(part[i], labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                all_loss += loss.item() * inputs.size(0)
                all_corrects += torch.sum(preds == labels.data)

            epoch_loss = all_loss / datas[str(phase)+"_size"]
            epoch_acc = all_corrects.double() / datas[str(phase)+"_size"]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            log_file.write('{} epoch : {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, phase,epoch_loss, epoch_acc) + '\n')

            if phase == 'train':
                last_model_wts = model.state_dict()
                if epoch % 10 == 9:
                    save_network(model, epoch, stage)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    save_network(model, 'best', stage)
    model.load_state_dict(last_model_wts)
    save_network(model, 'last', stage)
    return model
#*************************** trained model **************************#
######################################################################

def get_net(is_parallel, net):
    return net.module if is_parallel else net
######################################################################
#*************************** PCB train ******************************#
#****************** Step1 : train the PCB model *********************#
def pcb_train(model, criterion, log_file, stage, num_epoch):
    ignored_params = list(map(id, get_net(is_parallel_train, model).classifiers.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, get_net(is_parallel_train, model).parameters())
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.01},
        {'params': get_net(is_parallel_train, model).classifiers.parameters(), 'lr': 0.1},
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
    model = _train(model, criterion, optimizer_ft, exp_lr_scheduler, log_file, stage, num_epochs=num_epoch)
    return model
#*************************** trained model **************************#
######################################################################

######################################################################
#*************************** RPP train ******************************#
#**************** Setp 2&3: train the rpp layers ********************#
def rpp_train(model, criterion, log_file, stage, num_epoch):
    optimizer_ft = optim.SGD(
        get_net(is_parallel_train, model).avgpool.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9, nesterov=True)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)
    model = _train(model, criterion, optimizer_ft, exp_lr_scheduler, log_file, stage, num_epochs=num_epoch)
    return model
#*************************** trained model **************************#
######################################################################

######################################################################
#*************************** full train *****************************#
#**************** Step 4: train the whole net ***********************#
def full_train(model, criterion, log_file, stage, num_epoch):
    ignored_params = list(map(id, get_net(is_parallel_train, model).classifiers.parameters()))
    ignored_params += list(map(id, get_net(is_parallel_train, model).avgpool.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, get_net(is_parallel_train, model).parameters())
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.001},
        {'params': get_net(is_parallel_train, model).classifiers.parameters(), 'lr': 0.01},
        {'params': get_net(is_parallel_train, model).avgpool.parameters(), 'lr': 0.01},
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)
    model = _train(model, criterion, optimizer_ft, exp_lr_scheduler, log_file, stage, num_epochs=num_epoch)
    return model
#*************************** trained model **************************#
######################################################################

######################################################################
#*************************** main train *****************************#
def main_train():
    f = open(args.save_dir + 'train_log.txt', 'w')

    copyfile('./train.py', args.save_dir + '/train.py')
    copyfile('./model.py', args.save_dir + '/model.py')
    
    model = PCB(len(datas["class"]))
    
    if gpu:
        model = model.cuda()
    if is_parallel_train:
        model = nn.DataParallel(model, device_ids=gpu_ids)

    criterion = nn.CrossEntropyLoss()

    model = pcb_train(model, criterion, f, "PCB", 60)
    
    if args.RPP:
        model = get_net(is_parallel_train, model).convert_to_rpp()
        
        if use_gpu:
            model = model.cuda()
        if is_parallel_train:
            model = nn.DataParallel(model, device_ids=gpu_ids)

        model = rpp_train(model, criterion, f, "RPP", 5)
        model = full_train(model, criterion, f, "full", 10)
    
    f.close()
#*************************** trained model **************************#
######################################################################

main_train()