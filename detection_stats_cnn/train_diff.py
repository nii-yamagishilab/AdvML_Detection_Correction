"""
Copyright (c) 2022, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------

This script is for training the CNN detector using differences features
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
from tqdm import tqdm
import argparse
from sklearn import metrics
import model
from my_dataset import AdvMLDataset


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default ='../dataset', help='path to root dataset')
parser.add_argument('--train_set', default ='classifying_train.csv', help='train set')
parser.add_argument('--val_set', default ='classifying_val.csv', help='validation set')
# parser.add_argument('--train_set', default ='classifying_train_targeted.csv', help='train set')
# parser.add_argument('--val_set', default ='classifying_val_targeted.csv', help='validation set')
# parser.add_argument('--train_set', default ='classifying_train_nontargeted.csv', help='train set')
# parser.add_argument('--val_set', default ='classifying_val_nontargeted.csv', help='validation set')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=64, help='batch size')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--resume', type=int, default=0, help="choose a epochs to resume from (0 to train from scratch)")
parser.add_argument('--outf', default='chkpt_diff', help='folder to output model checkpoints')
# parser.add_argument('--outf', default='chkpt_diff_targeted', help='folder to output model checkpoints')
# parser.add_argument('--outf', default='chkpt_diff_nontargeted', help='folder to output model checkpoints')
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

    my_model = model.Classifer_Diff()
    my_model.train(mode=True)

    model_loss = nn.CrossEntropyLoss()

    optimizer = Adam(my_model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    if opt.resume > 0:
        my_model.load_state_dict(torch.load(os.path.join(opt.outf,'model_' + str(opt.resume) + '.pt')))
        my_model.train(mode=True)
        optimizer.load_state_dict(torch.load(os.path.join(opt.outf,'optim_' + str(opt.resume) + '.pt')))

        if opt.gpu_id >= 0:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(opt.gpu_id)

    if opt.gpu_id >= 0:
        my_model.cuda(opt.gpu_id)
        model_loss.cuda(opt.gpu_id)

    dataset_train = AdvMLDataset(data_dir=opt.dataset, data_list=opt.train_set)
    assert dataset_train
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

    dataset_val = AdvMLDataset(data_dir=opt.dataset, data_list=opt.val_set)
    assert dataset_val
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))


    for epoch in range(opt.resume+1, opt.niter+1):
        count = 0
        loss_train = 0
        loss_test = 0

        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)

        for stats_data, labels_data in tqdm(dataloader_train):
            img_label = labels_data.numpy().astype(np.float)
            optimizer.zero_grad()

            if opt.gpu_id >= 0:
                stats_data = stats_data.cuda(opt.gpu_id)
                labels_data = labels_data.cuda(opt.gpu_id)

            input_v = Variable(stats_data)
            classes = my_model(input_v)

            loss_dis = model_loss(classes, Variable(labels_data, requires_grad=False))
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
        torch.save(my_model.state_dict(), os.path.join(opt.outf, 'model_%d.pt' % epoch))
        torch.save(optimizer.state_dict(), os.path.join(opt.outf, 'optim_%d.pt' % epoch))

        my_model.eval()

        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)

        count = 0

        for stats_data, labels_data in dataloader_val:
            img_label = labels_data.numpy().astype(np.float)

            if opt.gpu_id >= 0:
                stats_data = stats_data.cuda(opt.gpu_id)
                labels_data = labels_data.cuda(opt.gpu_id)

            input_v = Variable(stats_data)

            classes = my_model(input_v)

            loss_dis = model_loss(classes, Variable(labels_data, requires_grad=False))
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
        my_model.train(mode=True)

    text_writer.close()
