import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import random


# La creazione dell'oggetto makeDataset usufruisce di questa gen split, che semplicemente attraversa tutte
# le directory e si va a prendere tutti i video per ogni classe, a cui assegna una label unica (class_id), salvandosi così
# in Dataset il path dei direttori dei singoli video, in labels tutte le labels dei video e in numFrames il numero di frame per ogni video
# ogni video è ovviamente identificato da un unico indice
def gen_split(root_dir, stackSize, train):
    Dataset = []
    Labels = []
    NumFrames = []
    ActionLabels = {}
    root_dire = os.path.join(root_dir, 'processed_frames2')
    for dir_user in sorted(os.listdir(root_dire)):
        class_id = 0
        dir1 = os.path.join(root_dire, dir_user)
        print(f'Sono in {dir1}')
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
                    print(f'Sono in {dir2}')
                    if os.path.isfile(dir2):
                      continue
                    insts = sorted(os.listdir(dir2))
                    if insts != []:
                      for inst in insts:
                          inst_dir = os.path.join(dir2, inst)
                          inst_dir = os.path.join(inst_dir, 'rgb')
                          numFrames = len(glob.glob1(inst_dir, '*.png'))
                          #print(f'Numero frames: {numFrames}')
                          if numFrames >= stackSize:
                              #print('Lo salvo!')
                              Dataset.append(inst_dir)
                              Labels.append(class_id)
                              NumFrames.append(numFrames)
                    class_id += 1
        else:
            if dir_user == 'S2':
                for target in sorted(os.listdir(dir1)):
                    print(target)
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
                              Dataset.append(inst_dir)
                              Labels.append(class_id)
                              NumFrames.append(numFrames)
                    class_id += 1
    return Dataset, Labels, NumFrames


class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform=None, seqLen=20,
                 train=True, mulSeg=False, numSeg=1, fmt='.png'):
        self.images, self.labels, self.numFrames = gen_split(root_dir, 5, train)  # vedi sopra
        self.spatial_transform = spatial_transform  # transformation di data augmentation
        self.train = train  # se è train o val
        self.mulSeg = mulSeg  # identifica se ci sono sequenze multiple, ma da quel che ho visto è sempre a false
        self.numSeg = numSeg  # numero sequenze in caso la precedente sia true
        self.seqLen = seqLen  # lunghezza della sequenza
        self.fmt = fmt  # stringa che indica il formato delle immagine, sempre .jpg

    def __len__(self):
        return len(self.images)

    # Quando viene preso un video, questo viene aperto e ogni singolo frame subisce una transformation e
    # una conversione in rgb; tutti i frame aperti vengono inseriti in un altro tensore (inqSeq) che a questo
    # punto intuisco diventi una matrice n-dimensionale. Viene restituito questo nuovo tensore e la relativa label
    def __getitem__(self, idx):
        vid_name = self.images[idx]
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        inpSeq = []
        self.spatial_transform.randomize_parameters()
        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
            fl_name = vid_name + '/' + 'rgb' + str(int(np.floor(i))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            inpSeq.append(self.spatial_transform(img.convert('RGB')))
        inpSeq = torch.stack(inpSeq, 0)
        return inpSeq, label
