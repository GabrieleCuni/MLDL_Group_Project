import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import random

#La creazione dell'oggetto makeDataset usufruisce di questa gen split, che semplicemente attraversa tutte
#le directory e si va a prendere tutti i video per ogni classe, a cui assegna una label unica (class_id), salvandosi così
#in Dataset il path dei direttori dei singoli video, in labels tutte le labels dei video e in numFrames il numero di frame per ogni video
#ogni video è ovviamente identificato da un unico indice
def gen_split(root_dir, stackSize):
    Dataset = []
    Labels = []
    NumFrames = []
    root_dir = os.path.join(root_dir, 'frames')
    for dir_user in sorted(os.listdir(root_dir)):
        class_id = 0
        dir = os.path.join(root_dir, dir_user)
        for target in sorted(os.listdir(dir)):
            dir1 = os.path.join(dir, target)
            insts = sorted(os.listdir(dir1))
            if insts != []:
                for inst in insts:
                    inst_dir = os.path.join(dir1, inst)
                    numFrames = len(glob.glob1(inst_dir, '*.jpg'))
                    if numFrames >= stackSize:
                        Dataset.append(inst_dir)
                        Labels.append(class_id)
                        NumFrames.append(numFrames)
            class_id += 1
    return Dataset, Labels, NumFrames

class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform=None, seqLen=20,
                 train=True, mulSeg=False, numSeg=1, fmt='.jpg'):

        self.images, self.labels, self.numFrames = gen_split(root_dir, 5) # vedi sopra
        self.spatial_transform = spatial_transform #transformation di data augmentation
        self.train = train #se è train o val
        self.mulSeg = mulSeg #identifica se ci sono sequenze multiple, ma da quel che ho visto è sempre a false
        self.numSeg = numSeg #numero sequenze in caso la precedente sia true
        self.seqLen = seqLen #lunghezza della sequenza
        self.fmt = fmt #stringa che indica il formato delle immagine, sempre .jpg

    def __len__(self):
        return len(self.images)
	
	#Quando viene preso un video, questo viene aperto e ogni singolo frame subisce una transformation e
	#una conversione in rgb; tutti i frame aperti vengono inseriti in un altro tensore (inqSeq) che a questo
	#punto intuisco diventi una matrice n-dimensionale. Viene restituito questo nuovo tensore e la relativa label
    def __getitem__(self, idx):
        vid_name = self.images[idx]
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        inpSeq = []
        self.spatial_transform.randomize_parameters()
        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
            fl_name = vid_name + '/' + 'image_' + str(int(np.floor(i))).zfill(5) + self.fmt
            img = Image.open(fl_name)
            inpSeq.append(self.spatial_transform(img.convert('RGB')))
        inpSeq = torch.stack(inpSeq, 0)
        return inpSeq, label
