import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import glob
import sys


def gen_split(root_dir, stackSize, phase):
    DatasetX = []
    DatasetY = []
    ActionLabels ={}
    Labels = []
    NumFrames = []
    root_dire = os.path.join(root_dir, 'flow_x_processed')
    for dir_user in sorted(os.listdir(root_dire)):
        class_id = 0
        dir1 = os.path.join(root_dire, dir_user)
        print(f'Sono in {dir1}')
        if os.path.isfile(dir1):
            continue
        if phase == 'train':
            if dir_user == 'S1' or dir_user == 'S3' or dir_user == 'S4':
                for target in sorted(os.listdir(dir1)):
                    if target in ActionLabels.keys():
                        class_id = ActionLabels[target]
                    else:
                        ActionLabels[target] = class_id
                        dir2 = os.path.join(dir1, target)
                        print(f'Sono in {dir2}')
                        if os.path.isfile(dir2):
                            continue
                        insts = sorted(os.listdir(dir2))
                        if insts != []:
                            for inst in insts:
                                inst_dir = os.path.join(dir2, inst)
                                inst_dir = os.path.join(inst_dir, 'rgb')
                                numFrames = len(glob.glob1(inst_dir, '*.png'))
                                if numFrames >= stackSize:
                                    DatasetX.append(inst_dir)
                                    DatasetY.append(inst_dir.replace('flow_x_processed', 'flow_y_processed'))
                                    Labels.append(class_id)
                                    NumFrames.append(numFrames)
                        class_id += 1
            else:
                if dir_user == 'S2':
                    for target in sorted(os.listdir(dir1)):
                        if target in ActionLabels.keys():
                            class_id = ActionLabels[target]
                        else:
                            ActionLabels[target] = class_id
                        dir2 = os.path.join(dir1, target)
                        if os.path.isfile(dir2):
                            continue
                        insts = sorted(os.listdir(dir2))
                        if insts != []:
                            for inst in insts:
                                inst_dir = os.path.join(dir2, inst)
                                numFrames = len(glob.glob1(inst_dir, '*.png'))
                                if numFrames >= stackSize:
                                    DatasetX.append(inst_dir)
                                    DatasetY.append(inst_dir.replace('flow_x_processed', 'flow_y_processed'))
                                    Labels.append(class_id)
                                    NumFrames.append(numFrames)
                        class_id += 1



    return DatasetX, DatasetY, Labels, NumFrames


class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform=None, sequence=False, stackSize=5,
                 train=True, numSeg = 1, fmt='.png', phase='train'):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.imagesX, self.imagesY, self.labels, self.numFrames = gen_split(root_dir, stackSize, phase)
        self.spatial_transform = spatial_transform
        self.train = train
        self.numSeg = numSeg
        self.sequence = sequence
        self.stackSize = stackSize
        self.fmt = fmt
        self.phase = phase

    def __len__(self):
        return len(self.imagesX)

    def __getitem__(self, idx):
        vid_nameX = self.imagesX[idx]
        vid_nameY = self.imagesY[idx]
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        inpSeqSegs = []
        self.spatial_transform.randomize_parameters()
        if self.sequence is True:
            if numFrame <= self.stackSize:
                frameStart = np.ones(self.numSeg)
            else:
                frameStart = np.linspace(1, numFrame - self.stackSize + 1, self.numSeg, endpoint=False)
            for startFrame in frameStart:
                inpSeq = []
                for k in range(self.stackSize):
                    i = k + int(startFrame)
                    fl_name = vid_nameX + '/flow_x_' + str(int(round(i))).zfill(5) + self.fmt
                    img = Image.open(fl_name)
                    inpSeq.append(self.spatial_transform(img.convert('L'), inv=True, flow=True))
                    # fl_names.append(fl_name)
                    fl_name = vid_nameY + '/flow_y_' + str(int(round(i))).zfill(5) + self.fmt
                    img = Image.open(fl_name)
                    inpSeq.append(self.spatial_transform(img.convert('L'), inv=False, flow=True))
                inpSeqSegs.append(torch.stack(inpSeq, 0).squeeze())
            inpSeqSegs = torch.stack(inpSeqSegs, 0)
            return inpSeqSegs, label
        else:
            if numFrame <= self.stackSize:
                startFrame = 1
            else:
                if self.phase == 'train':
                    startFrame = random.randint(1, numFrame - self.stackSize)
                else:
                    startFrame = np.ceil((numFrame - self.stackSize)/2)
            inpSeq = []
            for k in range(self.stackSize):
                i = k + int(startFrame)
                fl_name = vid_nameX + '/flow_x_' + str(int(round(i))).zfill(5) + self.fmt
                img = Image.open(fl_name)
                inpSeq.append(self.spatial_transform(img.convert('L'), inv=True, flow=True))
                # fl_names.append(fl_name)
                fl_name = vid_nameY + '/flow_y_' + str(int(round(i))).zfill(5) + self.fmt
                img = Image.open(fl_name)
                inpSeq.append(self.spatial_transform(img.convert('L'), inv=False, flow=True))
            inpSeqSegs = torch.stack(inpSeq, 0).squeeze(1)
            return inpSeqSegs, label#, fl_name