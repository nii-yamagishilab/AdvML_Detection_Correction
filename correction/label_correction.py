"""
Copyright (c) 2022, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------

This script is used to evaluate the ability of correcting labels of adversarial samples

real image list's format:
path,network,real,true_label,predicted_#1,predicted_#2,predicted_#3,predicted_#4,predicted_#5

adversarial image list's format:
path,network,attack,true_label,predicted_#1,predicted_#2,predicted_#3,predicted_#4,predicted_#5
"""

import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from io import BytesIO
from PIL import Image
from PIL import ImageFilter
import numpy as np
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default ='../dataset', help='path to dataset')
parser.add_argument('--adv_list', default ='../dataset/advml_list.csv', help='path to adversarial image list')
parser.add_argument('--real_list', default ='../dataset/real_list.csv', help='path to real image list')
parser.add_argument('--output', type=str, default='labels', help='output folder')
parser.add_argument('--device', type=str, default='cuda:0', help='device to use, e.g. cpu, cuda:0, cuda:1')

opt = parser.parse_args()
print(opt)

jpeg = [100,95,90,85,80,75,70,65,60,55,50,45,40,35,30,25]
scale = [0.75, 0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2, 1.25]

transform_fwd = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

vgg16 = models.vgg16(pretrained=True)
vgg16.eval()
vgg16.to(opt.device)

vgg19 = models.vgg19(pretrained=True)
vgg19.eval()
vgg19.to(opt.device)

resnet18 = models.resnet18(pretrained=True)
resnet18.eval()
resnet18.to(opt.device)

resnet50 = models.resnet50(pretrained=True)
resnet50.eval()
resnet50.to(opt.device)

def jpeg_compression(img, compression):
    img = img.copy()
    out_io = BytesIO()
    img.save(out_io, format='JPEG', quality=compression)
    out_io.seek(0)
    return Image.open(out_io)

def scale_filter(img, scale, original_size=224):
    img = img.copy()
    img_size = int(original_size * scale) 
    return img.resize((img_size, img_size))

def classifying(img, network):
    data = transform_fwd(img).unsqueeze_(dim=0)
    data = data.to(opt.device)
    res = network(data)
    res = torch.softmax(res, dim=1).cpu()

    _, top5_res = torch.sort(res, descending=True)
    return top5_res[0, 0:5].numpy()

if __name__ == "__main__":

    txtwtr = open(os.path.join(opt.output, 'label_correction.txt'), 'w')

    count_real = 0
    count_real_fixed = 0
    count_advml = 0
    count_advml_fixed = 0

    # real database

    with open(opt.real_list) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            img = Image.open(os.path.join(opt.dataset, row[0]))

            if img.getbands()[0] == 'L':
                img = img.convert('RGB')

            if row[1] == 'vgg16':
                network = vgg16
            elif row[1] == 'vgg19':
                network = vgg19
            elif row[1] == 'resnet18':
                network = resnet18
            elif row[1] == 'resnet50':
                network = resnet50

            labels = np.array([], dtype=int)

            for item in jpeg:
                img_tmp = jpeg_compression(img, item)
                top_5_tmp = classifying(img_tmp, network)
                labels = np.concatenate((labels, top_5_tmp), axis=0)

            for item in scale:
                img_tmp = scale_filter(img, item)
                top_5_tmp = classifying(img_tmp, network)
                labels = np.concatenate((labels, top_5_tmp), axis=0)

            unique_e, counts_e = np.unique(labels, return_counts=True)
            labels = np.stack((unique_e, counts_e), axis= 1)
            labels = labels[labels[:,1].argsort()]
            labels = np.flip(labels, axis=0)

            top_5 = labels[0:5, 0]

            count_real += 1
            if int(row[3]) in top_5:
                count_real_fixed += 1

    print('Count real: %d\n' % count_real)
    print('Count real fixed: %d\n' % count_real_fixed)
    print('Ratio: %.2f\n' % (count_real_fixed/count_real*100))

    txtwtr.write('Count real: %d\n' % count_real)
    txtwtr.write('Count real fixed: %d\n' % count_real_fixed)
    txtwtr.write('Ratio: %.2f\n' % (count_real_fixed/count_real*100))

    txtwtr.flush()

    # advml database

    with open(opt.adv_list) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            img = Image.open(os.path.join(opt.dataset, row[0]))

            if img.getbands()[0] == 'L':
                img = img.convert('RGB')

            if row[1] == 'vgg16':
                network = vgg16
            elif row[1] == 'vgg19':
                network = vgg19
            elif row[1] == 'resnet18':
                network = resnet18
            elif row[1] == 'resnet50':
                network = resnet50

            labels = np.array([], dtype=int)

            for item in jpeg:
                img_tmp = jpeg_compression(img, item)
                top_5_tmp = classifying(img_tmp, network)
                labels = np.concatenate((labels, top_5_tmp), axis=0)

            for item in scale:
                img_tmp = scale_filter(img, item)
                top_5_tmp = classifying(img_tmp, network)
                labels = np.concatenate((labels, top_5_tmp), axis=0)

            unique_e, counts_e = np.unique(labels, return_counts=True)
            labels = np.stack((unique_e, counts_e), axis= 1)
            labels = labels[labels[:,1].argsort()]
            labels = np.flip(labels, axis=0)

            top_5 = labels[0:5, 0]

            count_advml += 1
            if int(row[3]) in top_5:
                count_advml_fixed += 1

    print('Count advml: %d\n' % count_advml)
    print('Count advml fixed: %d\n' % count_advml_fixed)
    print('Ratio: %.2f\n' % (count_advml_fixed/count_advml*100))

    txtwtr.write('Count advml: %d\n' % count_advml)
    txtwtr.write('Count advml fixed: %d\n' % count_advml_fixed)
    txtwtr.write('Ratio: %.2f\n' % (count_advml_fixed/count_advml*100))

    txtwtr.flush()
    txtwtr.close()