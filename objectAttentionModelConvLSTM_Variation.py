import torch
import resnetMod
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from MyConvLSTMCell import *
from itertools import permutations, combinations
import random

def pairWiseCombinations(numFrame):
  l = list(combinations(np.linspace(0, numFrame, numFrame, endpoint=False, dtype=int), 2))
  return np.array(l)

class OrderPredictionNetwork(nn.Module):
  def __init__(self, numFrame=4, seqLen=7):
    super(OrderPredictionNetwork, self).__init__()
    self.combination = pairWiseCombinations(numFrame)
    self.permutation = [[0,1,2,3],[0,2,1,3],[0,2,3,1],[0,1,3,2],[0,3,1,2],[0,3,2,1],[1,0,2,3],[1,0,3,2],[1,2,0,3],[2,0,1,3],[2,0,3,1],[2,1,0,3]]
    self.fc6 = nn.Linear(7*7*512, 512*2 )
    self.fc7 = nn.Linear(512*4, 512)
    self.classifier = nn.Linear(512*len(self.combination), 12)
    self.seqLen = seqLen

  def forward(self, feature_conv_tens, permutationIndex):
    frameSequence = []
    for b in range(feature_conv_tens.size(0)):
      order = permutationIndex[b].item()
      x = feature_conv_tens[b].contiguous().view(feature_conv_tens[b].size(0), 512*7*7) # shape = [7 or 16, 512*7*7]
      x = self.fc6(x) # shape = [7 or 16, 512*2]
      frameIndexes = sorted( random.sample(range(self.seqLen), 4) )
      shuffle = self.permutation[order]
      newFrameIndexes = []
      for i in shuffle:
        newFrameIndexes.append(frameIndexes[i])
      frameSequence.append( torch.index_select(x, 0, torch.LongTensor(newFrameIndexes).cuda()))       
    feat_orders_shuffle = torch.stack(frameSequence, 0) # shape = [32, 4, 512*2]
    feat_temp = []
    for f1,f2 in self.combination:
      x = torch.index_select(feat_orders_shuffle, 1, torch.LongTensor([f1,f2]).cuda() )  #shape = [32, 2, 512 * 2]
      x = x.view(x.size(0), 512 * 4) # shape = [32, 512 *4]
      x = self.fc7(x) #shape = [32, 512]
      feat_temp.append(x)
    feat_temp = torch.stack(feat_temp, 0) #shape = [6, 32, 512]
    feat_temp = feat_temp.permute(1,0,2) #shape = [32, 6, 512]
    feat_temp = feat_temp.contiguous().view(feat_temp.size(0), 512 * len(self.combination))
    y = self.classifier(feat_temp)
    return y
    


class attentionModel(nn.Module):
    def __init__(self, num_classes=61, mem_size=512, regression=True, seqLen=7):
        super(attentionModel, self).__init__()
        self.regression = regression
        self.num_classes = num_classes
        self.resNet = resnetMod.resnet34(True, True)
		
        #MS net
        #if MsFlag == True:
        #self.relu = F.relu()
        self.convMS = nn.Conv2d(512, 100, kernel_size=1, padding=0)
        #view
        if regression == False:
            self.MSfc = nn.Linear(100*7*7, 2*7*7)
            #view
            self.m = nn.Softmax(2)
            self.orderPredictionNetwork = OrderPredictionNetwork(numFrame=4, seqLen=seqLen)
        else:
            self.MSfc = nn.Linear(100*7*7, 1*7*7)
        #else :
        
		
        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight
        self.lstm_cell = MyConvLSTMCell(512, mem_size)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)
        #self.MsFlag = MsFlag

    def forward(self, inputVariable, regression, permutationIndex):
        state = (Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()),
                 Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()))
        MSfeats = []
        feature_conv_list = []
        orderFeats = None

        for t in range(inputVariable.size(0)):
            logit, feature_conv, feature_convNBN = self.resNet(inputVariable[t])
            # feature_conv: torch.Size([32, 512, 7, 7])
            bz, nc, h, w = feature_conv.size()
            feature_conv1 = feature_conv.view(bz, nc, h*w)
            probs, idxs = logit.sort(1, True)
            class_idx = idxs[:, 0]
            cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
            attentionMAP = F.softmax(cam.squeeze(1), dim=1)
            attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)
            attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)

            #if self.MsFlag == True:
            feat = F.relu(attentionFeat)
            feat = self.convMS(feat)
            feat = feat.view(feat.size(0), -1)
            feat = self.MSfc(feat)
            if self.regression == False:
                feat = feat.view(feat.size(0), 7*7, 2)
                MSfeat = self.m(feat)
                feature_conv_list.append(feature_conv)
            else:
                MSfeat = feat.view(feat.size(0), 7*7)
            MSfeats.append(MSfeat)
            #else: # Order Prediction Self-supervized
                #feature_conv_list.append(feature_conv)
              
            state = self.lstm_cell(attentionFeat, state)
        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)

        #if self.MsFlag == True:
        MSfeats = torch.stack(MSfeats, 0)
        #else: # Order Prediction Self-supervized
        if regression == False:
            tensorOfFeatureConv = torch.stack(feature_conv_list, dim=0)  # shape = [ 7 or 16, 32, 512, 7, 7]
            tensorOfFeatureConv = tensorOfFeatureConv.permute(1,0,2,3,4) # shape = [32, 7 or 16, 512, 7, 7]
            orderFeats = self.orderPredictionNetwork(tensorOfFeatureConv, permutationIndex) # forward()  
        
        return feats, feats1, MSfeats, orderFeats






