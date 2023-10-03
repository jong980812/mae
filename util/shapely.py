import os
import PIL

from torchvision import datasets, transforms
from custom_transform import ThresholdTransform,AddNoise,DetachWhite

data_transform = transforms.Compose([
                transforms.Resize((256,256)),
                # transforms.RandomCrop((224,224)),
                # transforms.Grayscale(3),
                # transforms.RandomInvert(1),
                # transforms.RandomRotation((180,180)),
                # transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                # DetachWhite(30),
                # AddNoise(50),
                ThresholdTransform(250),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5],
                #                         std = [0.5,0.5,0.5])
])