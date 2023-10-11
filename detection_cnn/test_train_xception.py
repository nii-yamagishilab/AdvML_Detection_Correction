"""
Copyright (c) 2022, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------

Script for testing trained XceptionNet
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
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from my_dataset import AdvMLDataset
from xception import Xception

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default ='..', help='path to root dataset')
parser.add_argument('--test_set', default ='../dataset/test.csv', help='test set')
# parser.add_argument('--test_set', default ='../dataset/test_targeted.csv', help='test set')
# parser.add_argument('--test_set', default ='../dataset/test_nontargeted.csv', help='test set')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=48, help='batch size')
parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--outf', default='chkpt_train_xception', help='folder to output model checkpoints')
parser.add_argument('--id', type=int, default=81, help='checkpoint ID')
# parser.add_argument('--outf', default='chkpt_train_xception_targeted', help='folder to output model checkpoints')
# parser.add_argument('--id', type=int, default=91, help='checkpoint ID')
# parser.add_argument('--outf', default='chkpt_train_xception_nontargeted', help='folder to output model checkpoints')
# parser.add_argument('--id', type=int, default=132, help='checkpoint ID')

opt = parser.parse_args()
print(opt)

if __name__ == '__main__':

    text_writer = open(os.path.join(opt.outf, 'test.txt'), 'w')

    transform_fwd = transforms.Compose([
        transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    dataset_test = AdvMLDataset(data_dir=opt.dataset, data_list=opt.test_set, transform=transform_fwd)
    assert dataset_test
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))

    xception = Xception(num_classes=1000)
    xception.fc = nn.Linear(2048, 2)

    xception.load_state_dict(torch.load(os.path.join(opt.outf,'xception_' + str(opt.id) + '.pt')))
    xception.eval()

    if opt.gpu_id >= 0:
        xception.cuda(opt.gpu_id)


    ##################################################################################

    tol_label = np.array([], dtype=np.float)
    tol_pred = np.array([], dtype=np.float)
    tol_pred_prob = np.array([], dtype=np.float)

    count = 0
    loss_test = 0

    for img_data, labels_data in tqdm(dataloader_test):
        img_label = labels_data.numpy().astype(np.float)

        if opt.gpu_id >= 0:
            img_data = img_data.cuda(opt.gpu_id)
            labels_data = labels_data.cuda(opt.gpu_id)

        input_v = Variable(img_data)

        classes = xception(input_v)

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
