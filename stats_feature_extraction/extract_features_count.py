"""
Copyright (c) 2022, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------

This script is used to create classifying data using counting features

Output file format
Short form
<jpeg*5, gaussian*5, rotation*5, scale*5>,label

Full form
path,network,true_label,predicted_#1,predicted_#2,predicted_#3,predicted_#4,predicted_#5,<short form>
"""

import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from io import BytesIO
from random import shuffle
import csv
from PIL import Image
from PIL import ImageFilter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_real', default ='dataset/real_test.csv')
parser.add_argument('--input_advml', default ='dataset/advml_test.csv')
parser.add_argument('--output_short', type=str, default='dataset/classifying_count_test.csv')
parser.add_argument('--output_full', type=str, default='dataset/classifying_count_test_detail.csv')
parser.add_argument('--device', type=str, default='cuda:0', help='device to use, e.g. cpu, cuda:0, cuda:1')

opt = parser.parse_args()
print(opt)

jpeg = [100,95,90,85,80,75,70,65,60,55,50,45,40,35,30,25]
gaussian = [2,3,4,5]
rotation = [1,2,3,4,5,6,7,8]
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

txt_short = open(opt.output_short, 'w')
txt_full = open(opt.output_full, 'w')

def jpeg_compression(img, compression):
    img = img.copy()
    out_io = BytesIO()
    img.save(out_io, format='JPEG', quality=compression)
    out_io.seek(0)
    return Image.open(out_io)

def gaussian_filter(img, radius):
    img = img.copy()
    return img.filter(ImageFilter.GaussianBlur(radius))

def rotation_filter(img, angel):
    img = img.copy()
    return img.rotate(angel)

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
    return top5_res[0, 0:5]

def write_to_file(input_path, label):
     with open(input_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            
            txt_full.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,' % (row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]))
            img = Image.open(row[0])

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

            top_5 = classifying(img, network)

            count_1 = 0
            count_2 = 0
            count_3 = 0
            count_4 = 0
            count_5 = 0
            
            for item in jpeg:
                img_tmp = jpeg_compression(img, item)
                top_5_tmp = classifying(img_tmp, network)

                diff = top_5 - top_5_tmp
                diff[diff != 0] = 1

                if diff[0] == 0:
                    count_1 += 1
                if diff[1] == 0:
                    count_2 += 1
                if diff[2] == 0:
                    count_3 += 1
                if diff[3] == 0:
                    count_4 += 1
                if diff[4] == 0:
                    count_5 += 1

            txt_short.write('%d,%d,%d,%d,%d,' % (count_1, count_2, count_3,count_4,count_5))
            txt_full.write('%d,%d,%d,%d,%d,' % (count_1, count_2, count_3,count_4,count_5))

            count_1 = 0
            count_2 = 0
            count_3 = 0
            count_4 = 0
            count_5 = 0

            for item in gaussian:
                img_tmp = gaussian_filter(img, item)
                top_5_tmp = classifying(img_tmp, network)

                diff = top_5 - top_5_tmp
                diff[diff != 0] = 1

                if diff[0] == 0:
                    count_1 += 1
                if diff[1] == 0:
                    count_2 += 1
                if diff[2] == 0:
                    count_3 += 1
                if diff[3] == 0:
                    count_4 += 1
                if diff[4] == 0:
                    count_5 += 1

            txt_short.write('%d,%d,%d,%d,%d,' % (count_1, count_2, count_3,count_4,count_5))
            txt_full.write('%d,%d,%d,%d,%d,' % (count_1, count_2, count_3,count_4,count_5))

            count_1 = 0
            count_2 = 0
            count_3 = 0
            count_4 = 0
            count_5 = 0

            for item in rotation:
                img_tmp = rotation_filter(img, item)
                top_5_tmp = classifying(img_tmp, network)

                diff = top_5 - top_5_tmp
                diff[diff != 0] = 1

                if diff[0] == 0:
                    count_1 += 1
                if diff[1] == 0:
                    count_2 += 1
                if diff[2] == 0:
                    count_3 += 1
                if diff[3] == 0:
                    count_4 += 1
                if diff[4] == 0:
                    count_5 += 1

            txt_short.write('%d,%d,%d,%d,%d,' % (count_1, count_2, count_3,count_4,count_5))
            txt_full.write('%d,%d,%d,%d,%d,' % (count_1, count_2, count_3,count_4,count_5))

            count_1 = 0
            count_2 = 0
            count_3 = 0
            count_4 = 0
            count_5 = 0

            for item in scale:
                img_tmp = scale_filter(img, item)
                top_5_tmp = classifying(img_tmp, network)

                diff = top_5 - top_5_tmp
                diff[diff != 0] = 1

                if diff[0] == 0:
                    count_1 += 1
                if diff[1] == 0:
                    count_2 += 1
                if diff[2] == 0:
                    count_3 += 1
                if diff[3] == 0:
                    count_4 += 1
                if diff[4] == 0:
                    count_5 += 1

            txt_short.write('%d,%d,%d,%d,%d,' % (count_1, count_2, count_3,count_4,count_5))
            txt_full.write('%d,%d,%d,%d,%d,' % (count_1, count_2, count_3,count_4,count_5))

            txt_short.write('%d\n' % (label))
            txt_full.write('%d\n' % (label))

            txt_short.flush()
            txt_full.flush()


if __name__ == "__main__":

    write_to_file(opt.input_real, 0)
    write_to_file(opt.input_advml, 1)