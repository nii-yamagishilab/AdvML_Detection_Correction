"""
Copyright (c) 2022, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------

This script is used to evaluate nontarget adversarial machine learning attacks (default parameters)
on selected test items stored in datalist (usually top 5)

ouput file format:
path,network,attack,true_label,predicted_#1,predicted_#2,predicted_#3,predicted_#4,predicted_#5
"""

import os
import sys
import traceback
import torch
import numpy as np
from scipy.special import softmax
from imagenet import ImageNetDataset
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import foolbox
from foolbox.criteria import TopKMisclassification
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default ='../dataset', help='path to dataset folder')
parser.add_argument('--datalist', default ='data/resnet50_1000_top5.csv', help='path to the list of selected items from dataset')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--resize_size', type=int, default=256, help='the height / width of the resized image')
parser.add_argument('--cropped_size', type=int, default=224, help='the height / width of the cropped image')
parser.add_argument('--network', type=str, default='vgg16', help='network name, e.g. vgg16, vgg19, resnet18, resnet50')
parser.add_argument('--output', type=str, default='outputs/adversarial_list/untargeted', help='output folder')
parser.add_argument('--device', type=str, default='cuda:0', help='device to use, e.g. cpu, cuda:0, cuda:1')
parser.add_argument('--attack', type=str, default='gradient', help='attack name: gradient, fgsm, deepfool, newton, adef, slsqp, saliency, carlini_wagner')
parser.add_argument('--output_image', type=str, default='adv_images/untargeted/vgg16_gradient', help='empty: do not store image, otherwise: path to folder')

opt = parser.parse_args()
print(opt)

if __name__ == "__main__":

    # modify this file name to add more info right before '.csv'
    text_writer = open(os.path.join(opt.output, opt.network + '_' + opt.attack + '.csv'), 'w')

    # foolbox will apply data normalization later
    transform_fwd = transforms.Compose([
        transforms.Resize(opt.resize_size),
        transforms.CenterCrop(opt.cropped_size),
        transforms.ToTensor()
        ])

    datset = ImageNetDataset(opt.dataset, opt.datalist, transform_fwd)
    dataloader = torch.utils.data.DataLoader(datset, batch_size=1, shuffle=False, num_workers=int(opt.workers))

    network = getattr(models, opt.network)
    classifier = network(pretrained=True)
    classifier.eval()
    classifier.to(opt.device)

    data_len = len(dataloader.dataset)
    fail_count = 0
    top5_count = 0
    num_class = 1000
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    fmodel = foolbox.models.PyTorchModel(classifier, bounds=(0, 1), num_classes=num_class, preprocessing=(mean, std))

    for path, data, label in tqdm(dataloader):
        # foolbox works on numpy array, not Torch tensor
        data = data.squeeze_().numpy()
        label = label[0].numpy()
        path = path[0]

        # get the predicted label from classifier (top 1)
        # corresponding to prediction = classifier(data) but in NumPy version
        prediction = fmodel.predictions(data)
        prediction = softmax(prediction)
        prediction = np.asscalar(np.argmax(prediction))

        # the target will not be in top 5
        criterion = TopKMisclassification(k=5)

        # define the attack method
        if opt.attack == 'gradient':
            attack = foolbox.attacks.GradientAttack(fmodel, criterion)
        elif opt.attack == 'fgsm':
            attack = foolbox.attacks.FGSM(fmodel, criterion)
        elif opt.attack == 'deepfool':
            attack = foolbox.attacks.DeepFoolAttack(fmodel, criterion)
        elif opt.attack == 'newton':
            attack = foolbox.attacks.NewtonFoolAttack(fmodel, criterion)
        elif opt.attack == 'adef':
            attack = foolbox.attacks.ADefAttack(fmodel, criterion)
        elif opt.attack == 'slsqp':
            attack = foolbox.attacks.SLSQPAttack(fmodel, criterion)
        elif opt.attack == 'saliency':
            attack = foolbox.attacks.SaliencyMapAttack(fmodel, criterion)
        elif opt.attack == 'carlini_wagner':
            attack = foolbox.attacks.CarliniWagnerL2Attack(fmodel, criterion)

        # performe adversarial attack
        try:
            adversarial = attack(data, prediction)
        except AssertionError:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb) # Fixed format
            fail_count += 1
            continue

        if adversarial is None:
            fail_count += 1
            continue
        
        # get the predicted labels of created adversarial image
        prediction = fmodel.predictions(adversarial)
        prediction = softmax(prediction)
        top5_res = prediction.argsort()[-5:][::-1]

        filename = os.path.join(opt.output_image, path)

        if label in top5_res:
            top5_count += 1
        else:
            text_writer.write('%s,%s,%s,%d,%d,%d,%d,%d,%d\n' % (filename, opt.network, opt.attack, label, top5_res[0], top5_res[1], top5_res[2], top5_res[3], top5_res[4]))

        # unnormalize the adversarial image and save it
        if opt.output_image != '':
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
    
            adversarial = adversarial.transpose(1, 2, 0)
            adversarial = (adversarial*255).astype(np.uint8)
            adversarial = Image.fromarray(adversarial)
            # quality = 100 to keep its original quality
            adversarial.save(filename, quality=100)

        text_writer.flush()

    print('Could not perform adversarial ml: %d' % (fail_count))
    print('Top 5 accuracy: %.2f' % (float(top5_count)/data_len*100))

    text_writer.close()