import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import glob
import sys

#Analogamente al Dataset RGB, la creazione di un oggetto Dataset utilizza questo metodo con le dovute differenze:
#-attraversa tutti i percorsi dei flow 2d ("flow_x_processed and flow_y_processed → x and y Warp Flow data") 
#di ogni video per ogni classe
#-DatasetX memorizza il path relativo al flow_x di un determinato video 
#-DatasetY "         "   "    "       "  flow_y "  "  "             " 
#-Labels memorizza la classe relativa ad ogni video 
#-NumFrames memorizza il numero di frame di ogni video
#-Come in Dataset RGB, ogni video ha il suo indice 
def gen_split(root_dir, stackSize):
    DatasetX = []
    DatasetY = []
    Labels = []
    NumFrames = []
    root_dir = os.path.join(root_dir, 'flow_x')
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
                        DatasetX.append(inst_dir)
                        DatasetY.append(inst_dir.replace('flow_x', 'flow_y'))
                        Labels.append(class_id)
                        NumFrames.append(numFrames)
            class_id += 1
    return DatasetX, DatasetY, Labels, NumFrames


class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform=None, sequence=False, stackSize=5,
                 train=True, numSeg = 1, fmt='.jpg', phase='train'):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.imagesX, self.imagesY, self.labels, self.numFrames = gen_split(root_dir, stackSize) #memorizazione dati, vedi sopra
        self.spatial_transform = spatial_transform #trasformazione geometrica opzionale per data augmentation 
        self.train = train #di default settato a train, la modalità viene cambiata in phase
        self.numSeg = numSeg #numero sequenze, di deaflut a 1
        self.sequence = sequence #booleano se ho sequenze o meno, di default falso (vedi meglio, passaggio non chiaro)
        self.stackSize = stackSize #dimensione del mio stack di optical flow images 
        self.fmt = fmt #formato delle immagini, "jpeg"
        self.phase = phase #fase che permette di settare la modalità di train o test 

    def __len__(self):
        return len(self.imagesX)
    
#Quando viene preso un video, si vanno a valutare i dati flow per capire come fare lo stack di optical flow images 
#PREMESSA : non mi è ben chiara la questione delle sequenze. In base al fato se ho sequenze o meno cambia la scelta del 
#frame di partenza (cambia anche se sono in fase train)
#in sostanza: vience preso un video in termini di flow,  in base al frame di partenza e al numero di frame del flow interessato 
#viene prelevato un numero di frame uguale alla stack size
#ogni immagine scelta viene aperta, convertita in GreyScale ("L), se è flow_x viene invertita, se è flow_y no
#queste immagini trasformante vengo salvate nella lista di stacked images inpSeqSegs (stacked optical flow input)
#viene fatto un nuovo tensore a partire dalla lista di stacked images, con tutte le dimensioni di input uguali a 1 rimosse

    def __getitem__(self, idx):
        vid_nameX = self.imagesX[idx] #percoso flow_x del video il cui indice è idx
        vid_nameY = self.imagesY[idx] #percorso flow_y " " " " " " 
        label = self.labels[idx] # classe corrispondente
        numFrame = self.numFrames[idx] # num frames corrispondente 
        inpSeqSegs = [] #stacked optical flow input, il goal di tutto ciò è usare una stacked di 5 immagini come input
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
                    fl_name = vid_nameX + '/flow_x_' + str(int(round(i))).zfill(5) + '.jpg'
                    img = Image.open(fl_name)
                    inpSeq.append(self.spatial_transform(img.convert('L'), inv=True, flow=True))
                    # fl_names.append(fl_name)
                    fl_name = vid_nameY + '/flow_y_' + str(int(round(i))).zfill(5) + '.jpg'
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
                fl_name = vid_nameX + '/flow_x_' + str(int(round(i))).zfill(5) + '.jpg'
                img = Image.open(fl_name)
                inpSeq.append(self.spatial_transform(img.convert('L'), inv=True, flow=True))
                # fl_names.append(fl_name)
                fl_name = vid_nameY + '/flow_y_' + str(int(round(i))).zfill(5) + '.jpg'
                img = Image.open(fl_name)
                inpSeq.append(self.spatial_transform(img.convert('L'), inv=False, flow=True))
            inpSeqSegs = torch.stack(inpSeq, 0).squeeze(1)
            return inpSeqSegs, label#, fl_name
