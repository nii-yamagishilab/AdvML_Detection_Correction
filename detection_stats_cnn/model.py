"""
Copyright (c) 2022, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------

Script for the CNN model class using statistical features (counting and differences)
"""

import torch
from torch import nn

# counting: 20
# difference: 190

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)

class Classifer_Count(nn.Module):
    def __init__(self):
        super(Classifer_Count, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),

            nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            View(-1, 640),

            nn.Linear(640, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 2),
        )

        self.classifier.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(0.5, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x):
        return self.classifier(x)

class Classifer_Diff(nn.Module):
    def __init__(self):
        super(Classifer_Diff, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, stride=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),

            nn.Conv1d(8, 16, kernel_size=5, stride=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            View(-1, 1280),

            nn.Linear(1280, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 2),
        )

        self.classifier.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(0.5, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x):
        return self.classifier(x)