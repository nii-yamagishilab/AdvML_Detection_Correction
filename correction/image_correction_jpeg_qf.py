"""
Copyright (c) 2022, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------

This script is used to evaluate the ability of correcting adversarial images using JPEG compression
and its effects on natural images 

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
parser.add_argument('--output', type=str, default='images_qf', help='output folder')
parser.add_argument('--qf', type=int, help='JPEG compression quality factor')
parser.add_argument('--device', type=str, default='cuda:0', help='device to use, e.g. cpu, cuda:0, cuda:1')

opt = parser.parse_args()
print(opt)

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

    output_path = os.path.join(opt.output, str(opt.qf))

    if not os.path.exists(os.path.join(opt.output, str(opt.qf))):
        os.makedirs(os.path.join(opt.output, str(opt.qf)))

    txtwtr_stats = open(os.path.join(output_path, 'correction_stats.txt'), 'w')
    txtwtr_real = open(os.path.join(output_path, 'correction_real.csv'), 'w')
    txtwtr_advml = open(os.path.join(output_path, 'correction_advml.csv'), 'w')

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

            count_real += 1

            out_io = BytesIO()
            img.save(out_io, format='JPEG', quality=opt.qf)
            out_io.seek(0)
            corrected_img = Image.open(out_io)

            top_5 = classifying(corrected_img, network)

            if int(row[3]) in top_5:

                img_name = os.path.basename(row[0])
                img_name = img_name.split('.')[0] + '.JPEG'
                img_path = os.path.join(output_path, 'corrected_imgs', 'real', row[1])

                if not os.path.exists(img_path):
                    os.makedirs(img_path)

                img_path = os.path.join(img_path, img_name)
                img.save(img_path, format='JPEG', quality=opt.qf)
                
                txtwtr_real.write('%s,%s,%s,%s,%d\n' % (row[0], row[1], row[2], img_path, opt.qf))

                count_real_fixed += 1

    print('Count real: %d\n' % count_real)
    print('Count real fixed: %d\n' % count_real_fixed)
    print('Ratio: %.2f\n' % (count_real_fixed/count_real*100))

    txtwtr_stats.write('Count real: %d\n' % count_real)
    txtwtr_stats.write('Count real fixed: %d\n' % count_real_fixed)
    txtwtr_stats.write('Ratio: %.2f\n' % (count_real_fixed/count_real*100))

    txtwtr_stats.flush()

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

            count_advml += 1
            
            quality=str(opt.qf)

            out_io = BytesIO()
            img.save(out_io, format='JPEG', quality=opt.qf)
            out_io.seek(0)
            corrected_img = Image.open(out_io)

            top_5 = classifying(corrected_img, network)

            if int(row[3]) in top_5:

                img_name = os.path.basename(row[0])
                img_name = img_name.split('.')[0] + '.JPEG'
                img_path = os.path.join(output_path, 'corrected_imgs', 'advml', row[1] + '_' + row[2])

                if not os.path.exists(img_path):
                    os.makedirs(img_path)

                img_path = os.path.join(img_path, img_name)
                img.save(img_path, format='JPEG', quality=opt.qf)
                
                txtwtr_advml.write('%s,%s,%s,%s,%d\n' % (row[0], row[1], row[2], img_path, opt.qf))

                count_advml_fixed += 1

    print('Count advml: %d\n' % count_advml)
    print('Count advml fixed: %d\n' % count_advml_fixed)
    print('Ratio: %.2f\n' % (count_advml_fixed/count_advml*100))

    txtwtr_stats.write('Count advml: %d\n' % count_advml)
    txtwtr_stats.write('Count advml fixed: %d\n' % count_advml_fixed)
    txtwtr_stats.write('Ratio: %.2f\n' % (count_advml_fixed/count_advml*100))

    txtwtr_stats.flush()
    txtwtr_stats.close()