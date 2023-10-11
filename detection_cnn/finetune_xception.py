"""
Copyright (c) 2022, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------

Script for finetuning XceptionNet
xception-b5690688.pth can be found at:
http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth
"""

import os
import random
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import numpy as np
from torch.autograd import Variable
from torch.optim import Adam
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from sklearn import metrics
from my_dataset import AdvMLDataset
from xception import Xception


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default ='..', help='path to root dataset')
parser.add_argument('--train_set', default ='../dataset/train.csv', help='train set')
parser.add_argument('--val_set', default ='../dataset/val.csv', help='validation set')
# parser.add_argument('--train_set', default ='../dataset/train_targeted.csv', help='train set')
# parser.add_argument('--val_set', default ='../dataset/val_targeted.csv', help='validation set')
# parser.add_argument('--train_set', default ='../dataset/train_nontargeted.csv', help='train set')
# parser.add_argument('--val_set', default ='../dataset/val_nontargeted.csv', help='validation set')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=48, help='batch size')
parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--resume', type=int, default=0, help="choose a epochs to resume from (0 to train from scratch)")
parser.add_argument('--outf', default='chkpt_finetune_xception', help='folder to output model checkpoints')
# parser.add_argument('--outf', default='chkpt_finetune_xception_targeted', help='folder to output model checkpoints')
# parser.add_argument('--outf', default='chkpt_finetune_xception_nontargeted', help='folder to output model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

if __name__ == "__main__":

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.gpu_id >= 0:
        torch.cuda.manual_seed_all(opt.manualSeed)
        cudnn.benchmark = True

    if opt.resume > 0:
        text_writer = open(os.path.join(opt.outf, 'train.csv'), 'a')
    else:
        text_writer = open(os.path.join(opt.outf, 'train.csv'), 'w')

    xception = Xception(num_classes=1000)

    state_dict = torch.load('xception-b5690688.pth')
    for name, weights in state_dict.items():
        if 'pointwise' in name:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
    xception.load_state_dict(state_dict)
    xception.fc = nn.Linear(2048, 2)

    for param in xception.parameters():
        param.requires_grad = True

    xception.train(mode=True)

    xception_loss = nn.CrossEntropyLoss()

    optimizer = Adam(xception.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    if opt.resume > 0:
        xception.load_state_dict(torch.load(os.path.join(opt.outf,'xception_' + str(opt.resume) + '.pt')))
        xception.train(mode=True)
        optimizer.load_state_dict(torch.load(os.path.join(opt.outf,'optim_' + str(opt.resume) + '.pt')))

        if opt.gpu_id >= 0:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(opt.gpu_id)

    if opt.gpu_id >= 0:
        xception.cuda(opt.gpu_id)
        xception_loss.cuda(opt.gpu_id)

    transform_fwd = transforms.Compose([
        transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    dataset_train = AdvMLDataset(data_dir=opt.dataset, data_list=opt.train_set, transform=transform_fwd)
    assert dataset_train
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

    dataset_val = AdvMLDataset(data_dir=opt.dataset, data_list=opt.val_set, transform=transform_fwd)
    assert dataset_val
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))


    for epoch in range(opt.resume+1, opt.niter+1):
        count = 0
        loss_train = 0
        loss_test = 0

        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)

        for img_data, labels_data in tqdm(dataloader_train):
            img_label = labels_data.numpy().astype(np.float)
            optimizer.zero_grad()

            if opt.gpu_id >= 0:
                img_data = img_data.cuda(opt.gpu_id)
                labels_data = labels_data.cuda(opt.gpu_id)

            input_v = Variable(img_data)
            classes = xception(input_v)

            loss_dis = xception_loss(classes, Variable(labels_data, requires_grad=False))
            loss_dis_data = loss_dis.item()

            loss_dis.backward()
            optimizer.step()

            classes = torch.softmax(classes, dim=1)
            output_dis = classes.data.cpu().numpy()
            output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

            for i in range(output_dis.shape[0]):
                if output_dis[i,1] >= output_dis[i,0]:
                    output_pred[i] = 1.0
                else:
                    output_pred[i] = 0.0

            tol_label = np.concatenate((tol_label, img_label))
            tol_pred = np.concatenate((tol_pred, output_pred))

            loss_train += loss_dis_data
            count += 1


        acc_train = metrics.accuracy_score(tol_label, tol_pred)
        loss_train /= count

        ########################################################################

        # do checkpointing & validation
        torch.save(xception.state_dict(), os.path.join(opt.outf, 'xception_%d.pt' % epoch))
        torch.save(optimizer.state_dict(), os.path.join(opt.outf, 'optim_%d.pt' % epoch))

        xception.eval()

        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)

        count = 0

        for img_data, labels_data in dataloader_val:
            img_label = labels_data.numpy().astype(np.float)

            if opt.gpu_id >= 0:
                img_data = img_data.cuda(opt.gpu_id)
                labels_data = labels_data.cuda(opt.gpu_id)

            input_v = Variable(img_data)

            classes = xception(input_v)

            loss_dis = xception_loss(classes, Variable(labels_data, requires_grad=False))
            loss_dis_data = loss_dis.item()

            classes = torch.softmax(classes, dim=1)
            output_dis = classes.data.cpu().numpy()
            output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

            for i in range(output_dis.shape[0]):
                if output_dis[i,1] >= output_dis[i,0]:
                    output_pred[i] = 1.0
                else:
                    output_pred[i] = 0.0

            tol_label = np.concatenate((tol_label, img_label))
            tol_pred = np.concatenate((tol_pred, output_pred))

            loss_test += loss_dis_data
            count += 1

        acc_test = metrics.accuracy_score(tol_label, tol_pred)
        loss_test /= count

        print('[Epoch %d] Train loss: %.4f   acc: %.2f | Test loss: %.4f  acc: %.2f'
        % (epoch, loss_train, acc_train*100, loss_test, acc_test*100))

        text_writer.write('%d,%.4f,%.2f,%.4f,%.2f\n'
        % (epoch, loss_train, acc_train*100, loss_test, acc_test*100))

        text_writer.flush()
        xception.train(mode=True)

    text_writer.close()
