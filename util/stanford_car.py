from torch.utils.data import Dataset
import os
import PIL
from PIL import Image
import numpy as np
import json
import torch
import csv

class Stanford_car(Dataset):
    def __init__(self,transform, is_train, args):
        self.is_train = is_train
        self.data_path = os.path.join(args.data_path,('train' if is_train else 'test'))
        self.anno_path = os.path.join(args.anno_path, ('cars_train.csv' if is_train else 'cars_test.csv'))
        import pandas as pd
        # CSV 파일 경로
        # CSV 파일을 DataFrame으로 읽기
        df = pd.read_csv(self.anno_path,header=0,delimiter=',')
        self.transform = transform
        # 'path' 열을 self.image_paths로, 'lab' 열을 self.label_list로 설정
        self.image_list = df.values[:,0].tolist()
        self.label_list = df.values[:,-2].tolist()
    def __len__(self):
        return len(self.image_list)
    def __repr__(self) -> str:
        return 'Stanford Car' + ('Train' if self.is_train else 'Test')+ f': {len(self.image_list)}'
    def __getitem__(self, index):
        img_path = os.path.join(self.data_path,self.image_list[index])
        label = self.label_list[index]
        img = Image.open(img_path)
        if img.mode == 'L':
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label