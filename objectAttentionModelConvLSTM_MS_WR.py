import torch
import resnetMod
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from MyConvLSTMCell import *


class attentionModel(nn.Module):
    def __init__(self, num_classes=61, mem_size=512, regression=True):
        super(attentionModel, self).__init__()
        self.regression = regression
        self.num_classes = num_classes
        self.resNet = resnetMod.resnet34(True, True)
        self.convMS = nn.Conv2d(512, 100, kernel_size=1, padding=0)
        if regression == False:
            self.MSfc = nn.Linear(100*7*7, 2*7*7)
            self.m = nn.Softmax(2)
        else:
            self.MSfc = nn.Linear(100*7*7, 1*7*7)
		
		
        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight
        self.lstm_cell = MyConvLSTMCell(512, mem_size)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)

    def forward(self, inputVariable, regression):
        state = (Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()),
                 Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()))
        MSfeats = []
        for t in range(inputVariable.size(0)):
            logit, feature_conv, feature_convNBN = self.resNet(inputVariable[t])
            bz, nc, h, w = feature_conv.size()
            feature_conv1 = feature_conv.view(bz, nc, h*w)
            probs, idxs = logit.sort(1, True)
            class_idx = idxs[:, 0]
            cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
            attentionMAP = F.softmax(cam.squeeze(1), dim=1)
            attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)
            attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)
			
            feat = F.relu(attentionFeat)
            feat = self.convMS(feat)
            feat = feat.view(feat.size(0), -1)
            feat = self.MSfc(feat)
            if self.regression == False:
                feat = feat.view(feat.size(0), 7*7, 2)
                MSfeat = self.m(feat)
            else:
                MSfeat = feat.view(feat.size(0), 7*7)
            MSfeats.append(MSfeat)

            state = self.lstm_cell(attentionFeat, state)
        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)
        MSfeats = torch.stack(MSfeats, 0)
        return feats, feats1, MSfeats
