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


def build_dataset(is_train, args):
    if args.dataset == 'asd':
        transform = build_transform_asd(is_train, args)
    elif args.dataset == 'sketch_imagenet':    
        transform = build_transform_sketch_imagenet(is_train, args)
    elif args.dataset == 'tu_berlin':
        transform = build_transform_tu_berlin(is_train,args)
    elif args.dataset == 'asd_gray':
        transform = build_transform_asd_gray_scale(is_train,args)
    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform_asd(is_train, args):
    data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((384,128)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.96, 0.96, 0.96],
                                        std=[0.1, 0.1, 0.1])
                ]),
                'val': transforms.Compose([
                transforms.Resize((384,128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.96, 0.96, 0.96],
                                        std=[0.1, 0.1, 0.1])
                ])}
    # data_transforms = {
    #         'train': transforms.Compose([
    #             transforms.RandomResizedCrop(112),
    #             transforms.RandomHorizontalFlip(p=0.5),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    #             ]),
    #             'val': transforms.Compose([
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    #             ])}
    
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
                transforms.RandomRotation((-15,15)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.96, 0.96, 0.96],
                                        std=[0.1, 0.1, 0.1])
                ]),
                'val': transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.96, 0.96, 0.96],
                                        std=[0.1, 0.1, 0.1])
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
                transforms.Normalize(mean=[0.96],
                                        std=[0.1])
                ])}
    
    return data_transforms['train'] if is_train else data_transforms['val']#transforms.Compose(t)
