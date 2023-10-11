"""
Copyright (c) 2022, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------

This dataset class will load data from a list (data_list), which is stored in data_dir
"""

import os
import torch
import numpy as np
from random import randint
from PIL import Image
import csv

class AdvMLDataset():
    def __init__(self, data_dir, data_list):

        #parsing data_list
        reader_txt = csv.reader(open(os.path.join(data_dir, data_list), 'r'), delimiter=',')
        test_set = np.array(list(reader_txt)).astype('float')

        self.data = test_set[:, 0:-1]
        self.label = test_set[:, -1]

    def __getitem__(self, index):

        shape = self.data[index].shape
        dt = self.data[index].reshape(1, shape[0])
        lb = int(self.label[index])

        return (torch.FloatTensor(dt), lb)

    def __len__(self):
        return len(self.label)

    def name(self):
        return 'AdvML Dataset'
