# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from util.custom_transform import ThresholdTransform

def build_dataset(is_train, args):
    if args.dataset == 'asd_part_based' :
        mode = "train" if is_train else "val"
        return Part_based_dataset(args.data_path, args.json_path, mode, args.part_type)
    elif args.dataset == 'asd':
        transform = build_transform_asd(is_train, args)    
    elif args.dataset == 'DAPT':
        transform = build_transform_DAPT(is_train, args)
    elif args.dataset == 'pcb_asd':
        transform = build_transform_pcb_asd(is_train, args)
    elif args.dataset == 'sketch_imagenet':    
        transform = build_transform_sketch_imagenet(is_train, args)
    elif args.dataset == 'tu_berlin':
        transform = build_transform_tu_berlin(is_train,args)
    elif args.dataset == 'asd_gray':
        transform = build_transform_asd_gray_scale(is_train,args)
    elif args.dataset == 'binary':
        transform=build_transform_binary(is_train,args)
    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform_asd(is_train, args):
    data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224,224)),
                # transforms.Grayscale(3),
                # transforms.RandomInvert(1),
                # transforms.RandomRotation((-5,5)),
                # transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                # ThresholdTransform(10),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
                ]),
                'val': transforms.Compose([
                transforms.Resize((224,224)),
                # transforms.Grayscale(3),
                
                # transforms.RandomInvert(1),
                transforms.ToTensor(),
                # ThresholdTransform(10),
              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
                ])}
    
    return data_transforms['train'] if is_train else data_transforms['val'] # transforms.Compose(t)
def build_transform_DAPT(is_train, args):
    data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomRotation(degrees=(-10,10)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.95, 0.95, 0.95],
                                        std=[0.1, 0.1, 0.1])
                ]),
                'val': transforms.Compose([
                transforms.Resize((224,224)),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.95, 0.95, 0.95],
                                        std=[0.1, 0.1, 0.1])
                ])}
    
    return data_transforms['train'] if is_train else data_transforms['val'] # transforms.Compose(t) 
def build_transform_pcb_asd(is_train, args):
    data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((384,256)),
                transforms.ToTensor(),

                ]),
                'val': transforms.Compose([
                transforms.Resize((384,256)),
                transforms.ToTensor(),

                ])}
    
    return data_transforms['train'] if is_train else data_transforms['val']#transforms.Compose(t)
def build_transform_sketch_imagenet(is_train, args):
    data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.1,1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
               transforms.Normalize(mean=[0.96, 0.96, 0.96],
                                        std=[0.1, 0.1, 0.1])

                ]),
                'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
               transforms.Normalize(mean=[0.96, 0.96, 0.96],
                                        std=[0.1, 0.1, 0.1])

                ])}
    
    return data_transforms['train'] if is_train else data_transforms['val']#transforms.Compose(t)
def build_transform_tu_berlin(is_train, args):
    data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                ]),
                'val': transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                ])}
    return data_transforms['train'] if is_train else data_transforms['val']#transforms.Compose(t)


def build_transform_asd_gray_scale(is_train, args):
    data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((192,64)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Grayscale(1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.96],
                                        std=[0.1])
                ]),
                'val': transforms.Compose([
                transforms.Resize((192,64)),
                transforms.Grayscale(1),
                transforms.ToTensor(),
                ])}
    
    return data_transforms['train'] if is_train else data_transforms['val']#transforms.Compose(t)
def build_transform_binary(is_train, args):
    data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224,168)),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                ThresholdTransform(252),
                # transforms.Normalize(mean=[0.9],
                #                         std=[0.1])
                ]),
                'val': transforms.Compose([
                transforms.Resize((224,168)),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                ThresholdTransform(252),
                # transforms.Normalize(mean=[0.9],
                        # std=[0.1])
                ])}
    
    return data_transforms['train'] if is_train else data_transforms['val']#transforms.Compose(t)

import os
import json
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import json
import numpy as np

class Part_based_dataset(Dataset):
    def __init__(self, root_dir, json_dir, mode, part_type):
        is_train = True if mode == "train" else False
        self.transform = build_transform_asd(is_train, None)
        self.img_list, self.label_list = [],[]
        self.data_path = os.path.join(root_dir, mode) 
        self.class_ind = {'ASD': 0, 'TD': 1}
        self.ext = "jpg"
        self.json_path = json_dir
        self.part_type = part_type

        print("img_root_dir : ", self.data_path)
        print("json_root_dir : ", self.json_path)
        print("return_part_type : ", self.part_type)
        print("class_ind : ", self.class_ind)
        print("extension : ", self.ext)

        dataset = datasets.ImageFolder(self.data_path, self.transform)
        img_label_list = dataset.make_dataset(self.data_path, self.class_ind, extensions=self.ext)
        for img_label in img_label_list :
            self.img_list.append(img_label[0])
            self.label_list.append(img_label[1])

    def __len__(self):
        return len(self.img_list)

    def _crop_image(self, img, original_h, original_w, anns) :
        h, w = original_h, original_w
        p1, p2 = anns
        xmin, ymin = int(p1[0]), int(p1[1])
        xmax, ymax = int(p2[0]), int(p2[1])
        cropped_img = img.crop([xmin, ymin, xmax, ymax])   #* img.shape = (3, h, w)
        # cropped_img.save("/data/ahngeo11/mae-upstream/ex.png")
        return cropped_img

    def __getitem__(self, idx):
        totensor = transforms.ToTensor()
        
        image = Image.open(self.img_list[idx])  #* '/local_datasets/asd/compact_crop_trimmed_2/01/train/TD/B9-002-002.jpg'
        h, w = totensor(image).shape[1:]
        
        label = self.label_list[idx]

        json_name = self.img_list[idx].split('/')[-1].split('.')[0] + ".json"
        with open(os.path.join(self.json_path, json_name), 'r') as f :   
            part_anns = json.load(f)

        anns_dict = dict()
        for obj_ann in part_anns['shapes'] :
            anns_dict[obj_ann['label']] = obj_ann['points']
        
        #! implementation for 3 parts
        if self.part_type == 'head':
            img_head = self._crop_image(image, h, w, anns_dict['head'])
            img_head = self.transform(img_head)   #* transform -> float, (3, h, w)
            return img_head, label,self.img_list[idx]
        elif self.part_type == 'upper_body':
            img_upper_body = self._crop_image(image, h, w, anns_dict['upper_body'])
            img_upper_body = self.transform(img_upper_body)
            return img_upper_body, label,self.img_list[idx]
        elif self.part_type == 'lower_body':
            img_lower_body = self._crop_image(image, h, w, anns_dict['lower_body'])
            img_lower_body = self.transform(img_lower_body)
            return img_lower_body, label,self.img_list[idx]

        # return img_head, img_upper_body, img_lower_body, label