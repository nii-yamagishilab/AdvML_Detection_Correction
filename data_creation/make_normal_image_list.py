"""
Copyright (c) 2022, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------

This script is used to create a list of correct classified inputs (top 1 and top 5) from ImageNet 2012 val datasets
URL: http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar
valprep.sh must be run first to reorganize the data

Output: data list of correct classified inputs to opt.output folder
top 1 file format:
path,network,"real",true_label

top 5 file format:
path,network,"real",true_label,predicted_#1,predicted_#2,predicted_#3,predicted_#4,predicted_#5
"""

import os
import torch
import numpy as np
from folder import ImageFolder
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default ='../dataset', help='path to dataset folder')
parser.add_argument('--datatype', default ='ILSVRC2012_img_val', help='dataset type')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--batch_size', type=int, default=50, help='batch size')
parser.add_argument('--resize_size', type=int, default=256, help='the height / width of the resized image')
parser.add_argument('--cropped_size', type=int, default=224, help='the height / width of the cropped image')
parser.add_argument('--network', type=str, default='vgg16', help='network name, e.g. vgg16, vgg19, resnet18, resnet50')
parser.add_argument('--output', type=str, default='../dataset', help='output folder')
parser.add_argument('--device', type=str, default='cuda:0', help='device to use, e.g. cpu, cuda:0, cuda:1')

opt = parser.parse_args()
print(opt)

if __name__ == "__main__":

    txtwtr_top1 = open(os.path.join(opt.output, opt.network + '_top1.csv'), 'w')
    txtwtr_top5 = open(os.path.join(opt.output, opt.network + '_top5.csv'), 'w')

    transform_fwd = transforms.Compose([
        transforms.Resize(opt.resize_size),
        transforms.CenterCrop(opt.cropped_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    datset = ImageFolder(os.path.join(opt.dataset, opt.datatype), transform_fwd)
    dataloader = torch.utils.data.DataLoader(datset, batch_size=opt.batch_size, shuffle=False, num_workers=int(opt.workers))

    network = getattr(models, opt.network)
    classifier = network(pretrained=True)
    classifier.eval()
    classifier.to(opt.device)

    data_len = len(dataloader.dataset)
    top1_count = 0
    top5_count = 0

    for path, data, label in tqdm(dataloader):
        data = data.to(opt.device)
        res = classifier(data)
        res = torch.softmax(res, dim=1).cpu()

        top1_res = torch.argmax(res, dim=1)
        _, top5_res = torch.sort(res, descending=True)
        top5_res = top5_res[:,0:5]

        for i in range(label.shape[0]):
            if label[i] == top1_res[i]:
                txtwtr_top1.write('%s,%s,%s,%d\n' % (os.path.join(opt.datatype, path[i]), opt.network ,'real', label[i]))
                top1_count +=1


            if label[i] in top5_res[i]:
                txtwtr_top5.write('%s,%s,%s,%d,%d,%d,%d,%d,%d\n' % (os.path.join(opt.datatype, path[i]), opt.network ,'real', label[i], top5_res[i,0], top5_res[i,1], top5_res[i,2], top5_res[i,3], top5_res[i,4]))
                top5_count +=1

        txtwtr_top1.flush()
        txtwtr_top5.flush()

    print('Top 1 accuracy: %.2f' % (float(top1_count)/data_len*100))
    print('Top 5 accuracy: %.2f' % (float(top5_count)/data_len*100))

    txtwtr_top1.close()
    txtwtr_top5.close()