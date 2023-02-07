import os
import cv2
import random
import torch
import math
import pandas as pd
import torchvision
import numpy as np
from torch.utils.data import Dataset

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# ori_class: 0: car(小型車), 1: hov(大型車), 2: person(人), 3: motorcycle(機車)
# train_class: 1: hov(大型車), 2: person(人), 3: motorcycle(機車), 4: car(小型車)

def img_bbox_resize(img, resize: tuple, bbox=None):
    proportion_y, proportion_x = resize[1] / img.shape[0], resize[0] / img.shape[1]
    img = cv2.resize(img, resize)
    if bbox:
        bbox = [[round(x0*proportion_x), round(y0*proportion_y), round(x1*proportion_x), round(y1*proportion_y)] for x0, y0, x1, y1 in bbox]
        return img, bbox
    else:
        return img

class DroneDataset(Dataset):
    def __init__(self, split='train', resize=None, augmentation=None):
        self.augmentation = augmentation
        self.split = split
        self.resize = resize
        self.data_list = []

        if split == 'train':
            self.data_folder = 'train'
        else:
            self.data_folder = 'public'

        for file in os.listdir(os.path.join('dataset', self.data_folder)):
            if 'png' in file:
                if split == 'train':
                    with open(os.path.join('dataset', self.data_folder, file.replace('png', 'txt'))) as f:
                        if f.readlines():
                            self.data_list.append((os.path.join('dataset', self.data_folder, file),
                                                   os.path.join('dataset', self.data_folder, file.replace('png', 'txt'))))
                else:
                    self.data_list.append((os.path.join('dataset', self.data_folder, file), ''))

    def __getitem__(self, index):

        image_path, label_path = self.data_list[index]
        img = cv2.imread(image_path) / 255.  # H, W, C
        if self.augmentation is not None:
            img = self.augmentation(img)
        if self.data_folder == 'train':
            labels = []
            bboxes = []
            f = open(label_path, 'r')
            for line in f.readlines():
                cls, x0, y0, w, h = [int(x) for x in line.replace('\n', '').split(',')]
                cls = 4 if cls == 0 else cls
                labels.append(cls)
                bboxes.append([x0, y0, x0+w, y0+h])
            f.close()

        else:
            labels = None
            bboxes = None
        if self.resize:
            if self.data_folder == 'train':
                img, bboxes = img_bbox_resize(img, self.resize, bboxes)
            else:
                img = img_bbox_resize(img, self.resize)
        # img = torch.from_numpy(np.transpose(img.copy(), (2, 0, 1))).float()
        if self.data_folder == 'train':
            area = [w*h for x0, y0, w, h in bboxes]

            bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
            area = torch.as_tensor(area, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            iscrowd = torch.zeros((bboxes.shape[0],), dtype=torch.int64)

            target = {'boxes': bboxes, 'labels': labels, 'image_id': torch.tensor([index]), 'area': area, 'iscrowd': iscrowd}
        else:
            target = None

        img = torchvision.transforms.ToTensor()(img).type(torch.FloatTensor)
        return img, target, image_path.split('/')[-1]

    def class_proportion(self):
        count_c0 = 0
        count_c1 = 0
        count_c2 = 0
        count_c3 = 0
        for file in os.listdir(self.data_folder):
            if 'txt' in file:
                f = open(os.path.join(self.data_folder, file))
                for line in f.readlines():
                    cls = int(line.replace('\n', '').split(',')[0])
                    if cls == 0:
                        count_c0 += 1
                    if cls == 1:
                        count_c1 += 1
                    if cls == 2:
                        count_c2 += 1
                    if cls == 3:
                        count_c3 += 1
        return {'c0': count_c0, 'c1': count_c1, 'c2': count_c2, 'c3': count_c3}

    def __len__(self):
        return len(self.data_list)
