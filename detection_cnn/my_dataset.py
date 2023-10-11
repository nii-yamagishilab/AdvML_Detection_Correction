"""
Copyright (c) 2022, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------

Dataset class

Format:
path,network,attack,true_label,predicted_#1,predicted_#2,predicted_#3,predicted_#4,predicted_#5
"""

import os
import numpy as np
from random import randint
from PIL import Image
import csv

class AdvMLDataset():
    def __init__(self, data_dir, data_list, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.list = []

        #parsing data_list
        with open(data_list) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[2] == 'real':
                    self.list.append([row[0], 0])
                else:
                    self.list.append([row[0], 1])

    def __getitem__(self, index):

        path = os.path.join(self.data_dir, self.list[index][0])
        data = Image.open(path, mode='r').convert('RGB')

        if self.transform is not None:
            data = self.transform(data)

        return (data, int(self.list[index][1]))

    def __len__(self):
        return len(self.list)

    def name(self):
        return 'AdvML Dataset'
