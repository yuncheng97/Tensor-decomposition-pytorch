import torch
import os
from torch.autograd import Variable
from torchvision import models, utils
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from operator import itemgetter
import time
from datetime import timedelta
import tensorly as tl
import tensorly
from collections import OrderedDict
from itertools import chain
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torchstat import stat
from matplotlib import pyplot as plt

from decompositions import cp_decomposition_conv_layer, tucker_decomposition_conv_layer, estimate_ranks
from dataset_v2 import DogCat
# VGG16 based network for classifying between dogs and cats.
# After training this will be an over parameterized network,
# with potential to shrink it.

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AlexNetModel(torch.nn.Module):
    def __init__(self, model=None):
        super(AlexNetModel, self).__init__()

        model = models.alexnet(pretrained=True)
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2)
        )
        
    def forward(self,x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

class ModifiedVGG16Model(torch.nn.Module):
    def __init__(self, model=None):
        super(ModifiedVGG16Model, self).__init__()

        model = models.vgg16(pretrained=True)
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ModifiedResNet50Model(torch.nn.Module):
    def __init__(self, model=None):
        super(ModifiedResNet50Model, self).__init__()

        model = models.resnet50(pretrained=True)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc_inputs = model.fc.in_features
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_inputs, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# def ModifiedResNet50():
#     model = models.resnet50(pretrained=True)
#     # for param in model.parameters():
#     #     param.requires_grad = False
#     fc_inputs = model.fc.in_features
#     model.fc = nn.Sequential(
#         nn.Linear(fc_inputs, 256),
#         nn.ReLU(),
#         nn.Dropout(0.4),
#         nn.Linear(256, 2),
#         nn.LogSoftmax(dim=1)
#     )
#     return model


class Trainer:
    def __init__(self, train_loader, val_loader, model, optimizer, args, writer):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        # self.scheduler = scheduler
        self.args = args
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model.train()
        self.avg_meters = {
            'acc': AverageMeter(),
            'loss': AverageMeter()
        }
        self.writer = writer
    def train(self, epoches=1):
        for epoch in range(epoches):
            print("Epoch: [%d]", epoch)
            self.train_epoch()
            self.validate(epoch)
            # self.scheduler.step()

    def train_epoch(self):
        pbar = tqdm(total=len(self.train_loader))
        for i, (batch, label) in enumerate(self.train_loader):
            self.train_batch(batch.cuda(), label.cuda(), pbar)
        pbar.close()

    def train_batch(self, batch, label, pbar):
        self.model.zero_grad()
        input = Variable(batch)
        loss = self.criterion(self.model(input), Variable(label))
        loss.backward()
        self.optimizer.step()
        self.avg_meters['loss'].update(loss.item(), input.size(0))
        postfix = OrderedDict([('loss', self.avg_meters['loss'].avg)])
        pbar.set_postfix(postfix)
        pbar.update(1)

    def validate(self, epoch):
        self.model.cuda()
        self.model.eval()
        correct = 0
        total = 0
        total_time = 0
        start = time.time()
        for i, (batch, label) in enumerate(val_loader):
            batch = batch.cuda()
            t0 = time.time()
            output = model(Variable(batch)).cpu()
            t1 = time.time()
            total_time = total_time + (t1 - t0)
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)

        if self.args.fine_tune:
            self.writer.add_scalar('Accuracy', float(correct) / total, global_step=epoch)
        time_elapse = time.time() - start
        print("Accuracy :", float(correct) / total)
        print("CPU prediction time", float(total_time) / (i + 1), i + 1)
        print('inference time:', str(timedelta(seconds=time_elapse)))

        self.model.train()


def test(test_loader,args):
    #load model
    if args.model_path:
        if os.path.isfile(args.model_path):
            print("=> loading checkpoint '{}'".format(args.model_path))
            model = torch.load(args.model_path)
            model = model.cuda().eval()
        else:
            print("=> no checkpoint found at '{}'".format(args.model_path))
            
    results=[]
    for i, (images, path) in enumerate(test_loader):
        images = images.cuda(non_blocking=True)
        output=model(images)
        probability = nn.functional.softmax(output,dim=1)[:,1].detach().tolist()

        batch_results = [(path_.item(),probability_) for path_,probability_ in zip(path,probability)]

        results += batch_results
    write_csv(results,args.result)
    
    return results

def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)


def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: 
      tensor = tensor.view(n*c, -1, w, h)     
    elif c != 3: 
      tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))    
    plt.savefig('./visual/decompose_res_2.jpg')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--test", dest="test", action="store_true")
    parser.add_argument("--decompose", dest="decompose", action="store_true")
    parser.add_argument("--fine_tune", dest="fine_tune", action="store_true")
    parser.add_argument("--vis", dest="vis", action="store_true")
    parser.add_argument("--network", type = str, default = "VGG", help='VGG, Alex, Res')
    parser.add_argument("--result", type = str, default = "./outputs/VGG_result.csv", help='result file path')
    parser.add_argument("--model_path", type = str, default = "./models/VGG_model", help='VGG_model, Alex_model, Res_model')
    parser.add_argument("--decompose_model_path", type = str, default = "./models/VGG_tucker_model", help='VGG_decompose_model, Alex_decompose_model, Res_decompose_model')
    parser.add_argument("--train_path", type = str, default = "/home/yuncheng/tensor-decompositions/data/dogcat/train/train")
    parser.add_argument("--val_path", type = str, default = "/home/yuncheng/tensor-decompositions/data/dogcat/validation/validation")
    parser.add_argument("--test_path", type = str, default = "/home/yuncheng/tensor-decompositions/data/dogcat/test/test")
    parser.add_argument("--cp", dest="cp", action="store_true", \
        help="Use cp decomposition. uses tucker by default")
    parser.set_defaults(train=False)
    parser.set_defaults(decompose=False)
    parser.set_defaults(fine_tune=False)
    parser.set_defaults(cp=False)    
    parser.set_defaults(vis=False)  
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    writer_path = './tensorboard/Alex_cp/'
    writer = SummaryWriter(writer_path)
    args = get_args()
    tl.set_backend('pytorch')

    ######### dataset #########
    train_dataset = DogCat(root = args.train_path,train=True)
    val_dataset = DogCat(root = args.val_path,train=True)
    test_dataset = DogCat(root = args.test_path,test=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True,
        num_workers=4, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=True,
        num_workers=4, pin_memory=True, sampler=None
    )

    ######### training #########
    if args.train:
        model_name = args.network
        if model_name == 'VGG':
            model = ModifiedVGG16Model().cuda()
            print(model)
        elif model_name == 'Alex':
            model = AlexNetModel().cuda()
            print(model)
        elif model_name == 'Res':
            model = ModifiedResNet50Model().cuda()
            print(model)
        optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.99)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        # trainer = Trainer(args.train_path, args.val_path, model, optimizer)
        trainer = Trainer(train_loader, val_loader, model, optimizer, scheduler, args, writer)
        trainer.train(epoches = 20)
        print("Finished training.")
        torch.save(model, args.model_path)
        stat(model.cpu(),(3,224,224))

    ######### decompose #########
    elif args.decompose:
        # decompse resnet50 
        if args.model_path == './models/Res_model':
            model = torch.load(args.model_path).cuda()
            model.eval()
            model.cpu()
            mulfunc = (lambda x,y:x*y)
            for n, m in model.named_children():
                if n == 'avgpool':
                    break
                num_children = sum(1 for i in m.children())
                if num_children != 0:
                    # in a layer of resnet
                    layer = getattr(model, n)
                    # decomp every bottleneck
                    for i in range(num_children):
                        bottleneck = layer[i]
                        conv2 = getattr(bottleneck, 'conv2')
                        new_layers = tucker_decomposition_conv_layer(conv2) 
                        setattr(bottleneck, 'conv2', nn.Sequential(*new_layers))

                        del conv2
                        del bottleneck
                    del layer
                torch.save(model, args.decompose_model_path)
            stat(model, (3,224,224))
        # decompse vggnet16 abd alexnet
        else:
            model = torch.load(args.model_path).cuda()
            model.eval()
            model.cpu()
            N = len(model.features._modules.keys())
            print(N)
            for i, key in enumerate(model.features._modules.keys()):
                if i >= N - 2:
                    break
                if isinstance(model.features._modules[key], torch.nn.modules.conv.Conv2d):
                    conv_layer = model.features._modules[key]
                    if args.cp:
                        # rank = max(conv_layer.weight.data.numpy().shape)//3
                        rank = estimate_ranks(conv_layer)[0]
                        print('rank of the {%d}th layer: {%d} '%(i, rank))
                        decomposed = cp_decomposition_conv_layer(conv_layer, rank)
                    else:
                        decomposed = tucker_decomposition_conv_layer(conv_layer)

                    model.features._modules[key] = decomposed

                torch.save(model, args.decompose_model_path)
            stat(model, (3,224,224))

    ######### finetune #########
    elif args.fine_tune:
        base_model = torch.load(args.decompose_model_path)
        model = torch.nn.DataParallel(base_model)

        for param in model.parameters():
            param.requires_grad = True

        print(model)
        model.cuda()        

        if args.cp:
            optimizer = optim.SGD(model.parameters(), lr=0.000001)
        else:
            # optimizer = optim.SGD(chain(model.features.parameters(), \
            #     model.classifier.parameters()), lr=0.01)
            optimizer = optim.SGD(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        # trainer = Trainer(train_loader, val_loader, model, optimizer, scheduler, args, writer)
        trainer = Trainer(train_loader, val_loader, model, optimizer, args, writer)

        trainer.validate(epoch=0)
        model.cuda()
        model.train()
        trainer.train(epoches=20)
        print("Finished fine tuning.")
        model.eval()
        trainer.validate(epoch=21)

    ######### inference #########
    if args.test:
        test(test_loader,args)

    ######### visualization #########
    if args.vis:
        model = torch.load('./models/Res_decompose_model')
        layer = 0
    # if visualiing the original model, use this code:
        # filter = model.features[layer].weight.data.clone()

    # if visualing the decomposed model, use this code:
        # filter = model.features[layer][0].weight.data.clone()    

    # if visualiing the resnet model, use the following codes:
        bottleneck = model.layer1[layer]
        conv2 = getattr(bottleneck, 'conv2')
        filter = conv2[2].weight.data.clone()
        # filter = conv2.weight.data.clone()
        visTensor(filter, ch=0, allkernels=False)

        plt.axis('off')
        plt.ioff()
        plt.show()


    # if args.vis:
    #     a = 1
    #     def viz(module, input):
    #         global a
    #         x = input[0][0].cpu()
    #         for i in range(x.size()[0]):
    #             plt.xticks([])  #关闭x刻度
    #             plt.yticks([])  #关闭y刻度
    #             plt.axis('off') #关闭坐标轴
    #             plt.rcParams['figure.figsize'] = (20, 20) 
    #             # plt.rcParams['savefig.dpi'] = 240
    #             # plt.rcParams['figure.dpi'] = 240
    #             plt.imshow(x[i])
    #             plt.tight_layout()
    #             plt.savefig('./visual/111111VGG/'+str(a)+'_'+str(i)+'.jpg')
    #         a += 1
    #         # plt.show()

    #     model = torch.load('./models/VGG_model')
    #     dataiter = iter(val_loader)  # 迭代遍历图片
    #     images, labels = dataiter.next()

    #     for name, m in model.named_modules():
    #         if isinstance(m, torch.nn.Conv2d):
    #             m.register_forward_pre_hook(viz)

    #     model.eval()
    #     with torch.no_grad():
    #             model(images[2].unsqueeze(0).cuda())

