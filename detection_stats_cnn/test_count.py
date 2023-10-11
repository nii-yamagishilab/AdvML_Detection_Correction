"""
Copyright (c) 2022, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------

This script is for testing the CNN detector using counting features
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
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import model
from my_dataset import AdvMLDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default ='../dataset', help='path to root dataset')
parser.add_argument('--test_set', default ='classifying_count_test.csv', help='test set')
# parser.add_argument('--test_set', default ='classifying_count_test_targeted.csv', help='test set')
# parser.add_argument('--test_set', default ='classifying_count_test_nontargeted.csv', help='test set')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='batch size')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--outf', default='chkpt_count', help='folder to output model checkpoints')
parser.add_argument('--id', type=int, default=10, help='checkpoint ID')
# parser.add_argument('--outf', default='chkpt_count_targeted', help='folder to output model checkpoints')
# parser.add_argument('--id', type=int, default=21, help='checkpoint ID')
# parser.add_argument('--outf', default='chkpt_count_nontargeted', help='folder to output model checkpoints')
# parser.add_argument('--id', type=int, default=8, help='checkpoint ID')

opt = parser.parse_args()
print(opt)

if __name__ == '__main__':

    text_writer = open(os.path.join(opt.outf, 'test.txt'), 'w')

    dataset_test = AdvMLDataset(data_dir=opt.dataset, data_list=opt.test_set)
    assert dataset_test
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))

    my_model = model.Classifer_Count()
    my_model.load_state_dict(torch.load(os.path.join(opt.outf,'model_' + str(opt.id) + '.pt')))
    my_model.eval()

    if opt.gpu_id >= 0:
        my_model.cuda(opt.gpu_id)


    ##################################################################################

    tol_label = np.array([], dtype=np.float)
    tol_pred = np.array([], dtype=np.float)
    tol_pred_prob = np.array([], dtype=np.float)

    count = 0
    loss_test = 0

    for stats_data, labels_data in tqdm(dataloader_test):
        img_label = labels_data.numpy().astype(np.float)

        if opt.gpu_id >= 0:
            stats_data = stats_data.cuda(opt.gpu_id)
            labels_data = labels_data.cuda(opt.gpu_id)

        input_v = Variable(stats_data)

        classes = my_model(input_v)

        classes = torch.softmax(classes, dim=1)
        output_dis = classes.data.cpu()
        output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

        for i in range(output_dis.shape[0]):
            if output_dis[i,1] >= output_dis[i,0]:
                output_pred[i] = 1.0
            else:
                output_pred[i] = 0.0

        tol_label = np.concatenate((tol_label, img_label))
        tol_pred = np.concatenate((tol_pred, output_pred))
        
        pred_prob = torch.softmax(output_dis, dim=1)
        tol_pred_prob = np.concatenate((tol_pred_prob, pred_prob[:,1].data.numpy()))

        count += 1

    acc_test = metrics.accuracy_score(tol_label, tol_pred)
    loss_test /= count

    fpr, tpr, thresholds = roc_curve(tol_label, tol_pred_prob, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    print('[Epoch %d] Test acc: %.2f   EER: %.2f' % (opt.id, acc_test*100, eer*100))
    text_writer.write('%d,%.2f,%.2f\n'% (opt.id, acc_test*100, eer*100))

    text_writer.flush()
    text_writer.close()
