import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import random
from spatial_transforms import Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop, RandomHorizontalFlip, Binary

def gen_split(root_dir, stackSize, train):
    Dataset = []
    Labels = []
    NumFrames = []
    ActionLabels = {}
    root_dire = os.path.join(root_dir, 'processed_frames2')
    for dir_user in sorted(os.listdir(root_dire)):
        class_id = 0
        dir1 = os.path.join(root_dire, dir_user)
        if os.path.isfile(dir1):
          continue
        if train == True:
            if dir_user == 'S1' or dir_user == 'S3' or dir_user == 'S4':
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
                          inst_dir = os.path.join(inst_dir, 'rgb')
                          numFrames = len(glob.glob1(inst_dir, '*.png')) - len(glob.glob1(inst_dir, '*(1).png'))
                          if numFrames >= stackSize:
                              Dataset.append(inst_dir)
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
                          inst_dir = os.path.join(inst_dir, 'rgb')
                          numFrames = len(glob.glob1(inst_dir, '*.png')) - len(glob.glob1(inst_dir, '*(1).png'))
                          if numFrames >= stackSize:
                              Dataset.append(inst_dir)
                              Labels.append(class_id)
                              NumFrames.append(numFrames)
                    class_id += 1
    return Dataset, Labels, NumFrames


class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform=None, seqLen=20,
                 train=True, mulSeg=False, numSeg=1, fmt='.png', regression=True, numOrdClass=12):
        normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.images, self.labels, self.numFrames = gen_split(root_dir, 5, train)  # vedi sopra
        self.main_spatial_transform = spatial_transform  # transformation di data augmentation
        self.spatial_transform_rgb = Compose([self.main_spatial_transform, ToTensor(), normalize])
        if regression == False:
            self.spatial_transform_mmaps = Compose([self.main_spatial_transform, Scale(7), ToTensor(), Binary(0.4)])
        else:
            self.spatial_transform_mmaps = Compose([self.main_spatial_transform, Scale(7), ToTensor()])
        self.train = train  
        self.mulSeg = mulSeg  
        self.numSeg = numSeg  
        self.seqLen = seqLen  
        self.fmt = fmt  
        self.numOrdClass = numOrdClass

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        vid_name = self.images[idx]
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        inpSeq = []
        inpSeqMap = []
        self.main_spatial_transform.randomize_parameters()
        mmaps_vid_name = vid_name.replace('rgb', 'mmaps')

        labelOrder = np.random.randint(self.numOrdClass)

        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
            fl_name = vid_name + '/' + 'rgb' + str(int(np.floor(i))).zfill(4) + self.fmt
            fm_name = mmaps_vid_name + '/' + 'map' + str(int(np.floor(i))).zfill(4) + self.fmt
            if os.path.exists(fm_name) == False:
                fm_name = mmaps_vid_name + '/' + 'map' + str(int(np.floor(i+1))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            img_map = Image.open(fm_name)
            inpSeq.append(self.spatial_transform_rgb(img.convert('RGB')))
            inpSeqMap.append(self.spatial_transform_mmaps(img_map.convert('L')))
        inpSeq = torch.stack(inpSeq, 0)
        inpSeqMap = torch.stack(inpSeqMap, 0)
        return inpSeq, inpSeqMap, label, int(labelOrder)