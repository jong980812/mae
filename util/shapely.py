import os
import PIL
from PIL import Image
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
import itertools
from torchvision import datasets, transforms, models
from custom_transform import ThresholdTransform,AddNoise,DetachWhite
from einops import rearrange

import torchvision.models as models
model=models.efficientnet_b1(pretrained=True,progress=False)
model.classifier[1] = torch.nn.Linear(1280, 2)
import torchvision
# model=torchvision.models.resnet18()
# in_feat=model.fc.in_features
# model.fc=torch.nn.Linear(in_feat,2)
data_path='/data/datasets/asd/All_5split/01/val/TD/'
# data_path='/data/datasets/ai_hub_sketch_4way/01/val/m_w'
# data_path='/data/datasets/ai_hub/ai_hub_sketch_mw/01/val/w/'
import random
weight='/data/jong980812/project/mae/result_ver2/All_5split/binary_240/OUT/01/checkpoint-29.pth'
checkpoint = torch.load(weight, map_location='cpu')
print("Load pre-trained checkpoint from: %s" % weight)
checkpoint_model = checkpoint['model']
state_dict = model.state_dict()
msg = model.load_state_dict(checkpoint_model, strict=False)
model.eval()
print(msg)
class shapely_part(Dataset):
    def __init__(self, data_folder, json_folder, binary_thresholding=None, transform=None):
        self.json_folder = json_folder
        self.data_folder = data_folder
        self.binary_thresholding=binary_thresholding
        self.transform = transform
        self.image_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        self.json_paths = [image_path.split('/')[-1].split('.')[0] + ".json" for image_path in self.image_paths] #! Get json path from image paths.
        print(self.image_paths)
    def get_part_json(self, json_file_path, part_name):
        '''
        Get part dictionary from json path
        '''
        part_json = {}
        
        for part in part_name:
            part_json[part] = []
        with open(json_file_path, 'r') as f:
            boxes = json.load(f)['shapes']
            for box in boxes:
                part_json[box["label"]].append(box["points"])
    
        for key in part_json:#! 빈 애들은 None으로 처리해서 없다고 판단.
            if not part_json[key]:
                part_json[key] = None

        return part_json
    def get_coords(self, part):
        extracted_coordinates = []
        if part is None:
            return None
        elif len(part) == 1:
            # print(part[0][0])
            xmin, ymin = list(map(int,part[0][0]))
            xmax, ymax = list(map(int,part[0][1]))
            return [[xmin,ymin,xmax,ymax]]#아래 2일경우와 통일하기 위해 이중 리스트로 
        elif len(part) == 2:
            #! Eye, Ear, hand, foot -> These have 2 part, return list
            for a in part: 
                # print(a)
                xmin, ymin = list(map(int,a[0]))
                xmax, ymax = list(map(int,a[1]))
                extracted_coordinates.append([xmin,ymin,xmax,ymax])
            return extracted_coordinates
        else:
            exit(0)
    def get_white_image(self,size):
        return Image.new("RGB", size, (255, 255, 255))
    def get_empty_face(self,img, part_imgs, part_json):
        '''
        empty_face is face detached 'eye','nose','mouth','ear'
        '''
        head_json = part_json['head']
        head_coords = self.get_coords(head_json)
        head = part_imgs['head'][0]#!
        white_image = self.get_white_image(img.size)
        white_image.paste(head,head_coords[0])
        for part in ['eye','nose','mouth','ear']:
            if part_json[part] is not None:
              part_coords= self.get_coords(part_json[part])
              part_img = part_imgs[part]
              if part in ['eye','ear']:   
                  white_image.paste(self.get_white_image(part_img[0].size),part_coords[0])
                  white_image.paste(self.get_white_image(part_img[1].size),part_coords[1])
              else:
                  white_image.paste(self.get_white_image(part_img[0].size),part_coords[0])
                  
        return white_image 
    def get_empty_face(self,img, part_imgs, part_json):
        '''
        empty_face is face detached 'eye','nose','mouth','ear'
        '''
        head_json = part_json['head']
        head_coords = self.get_coords(head_json)
        head = part_imgs['head'][0]#!
        white_image = self.get_white_image(img.size)
        white_image.paste(head,head_coords[0])
        for part in ['eye','nose','mouth','ear']:
            if part_json[part] is not None:
              part_coords= self.get_coords(part_json[part])
              part_img = part_imgs[part]
              if part in ['eye','ear']:   
                  white_image.paste(self.get_white_image(part_img[0].size),part_coords[0])
                  white_image.paste(self.get_white_image(part_img[1].size),part_coords[1])
              else:
                  white_image.paste(self.get_white_image(part_img[0].size),part_coords[0])
        # white_image.show()
        return white_image
    def get_empty_lower_body(self,img, part_imgs, part_json):
        '''
        empty_lower_body detacched foot
        '''
        lower_body_json = part_json['lower_body']
        lower_body_coords = self.get_coords(lower_body_json)
        lower_body = part_imgs['lower_body'][0]#!
        white_image = self.get_white_image(img.size)
        white_image.paste(lower_body,lower_body_coords[0])
        if part_json["foot"] is not None:
            part_coords= self.get_coords(part_json["foot"])
            part_img = part_imgs["foot"] 
            white_image.paste(self.get_white_image(part_img[0].size),part_coords[0])
            white_image.paste(self.get_white_image(part_img[1].size),part_coords[1])
        
        return white_image.crop(lower_body_coords[0])
    def get_empty_upper_body(self,img, part_imgs, part_json):
        '''
        empty_lower_body detacched foot
        '''
        upper_body_json = part_json['upper_body']
        upper_body_coords = self.get_coords(upper_body_json)
        upper_body = part_imgs['upper_body'][0]#!
        white_image = self.get_white_image(img.size)
        white_image.paste(upper_body,upper_body_coords[0])
        if part_json["hand"] is not None:
            part_coords= self.get_coords(part_json["hand"])
            part_img = part_imgs["hand"] 
            white_image.paste(self.get_white_image(part_img[0].size),part_coords[0])
            white_image.paste(self.get_white_image(part_img[1].size),part_coords[1])
        # white_image.crop(upper_body_coords[0]).show()
        return white_image.crop(upper_body_coords[0])
    
    def create_new_images(self,img, binary_combination, part_imgs,part_json):
        #! Making New images
        original_img = img
        empty_face_active, eye_active, nose_active, ear_active, mouth_active, hand_active, foot_active = binary_combination
        # New white image

        new_image = self.get_white_image(original_img.size)
        if empty_face_active:
            new_image.paste(part_imgs["empty_face"][0],(0,0))
        # print(part_json['lower_body'][0])
        # print(part_imgs["empty_lower_body"][0].size,self.get_coords(part_json['lower_body'])[0] )
        new_image.paste(part_imgs["empty_lower_body"][0], self.get_coords(part_json['lower_body'])[0])  # 원하는 위치에 붙임
        new_image.paste(part_imgs["empty_upper_body"][0], self.get_coords(part_json['upper_body'])[0])  # 원하는 위치에 붙임
        # 각 파트 이미지를 읽어와서 새로운 이미지에 붙임
        if eye_active and (part_json["eye"] is not None):
            new_image.paste(part_imgs["eye"][0], self.get_coords(part_json['eye'])[0])  # 원하는 위치에 붙임
            new_image.paste(part_imgs["eye"][1], self.get_coords(part_json['eye'])[1])  # 원하는 위치에 붙임 
        if nose_active and (part_json["nose"] is not None):
            new_image.paste(part_imgs["nose"][0], self.get_coords(part_json['nose'])[0])  # 원하는 위치에 붙임 
        if ear_active and (part_json["ear"] is not None):
            new_image.paste(part_imgs["ear"][0], self.get_coords(part_json['ear'])[0])  # 원하는 위치에 붙임 
            new_image.paste(part_imgs["ear"][1], self.get_coords(part_json['ear'])[1])  # 원하는 위치에 붙임 
        if mouth_active and (part_json["mouth"] is not None):
            new_image.paste(part_imgs["mouth"][0], self.get_coords(part_json['mouth'])[0])  # 원하는 위치에 붙임 
        if hand_active and (part_json["hand"] is not None):
            new_image.paste(part_imgs["hand"][0], self.get_coords(part_json['hand'])[0])  # 원하는 위치에 붙임 
            new_image.paste(part_imgs["hand"][1], self.get_coords(part_json['hand'])[1])  # 원하는 위치에 붙임 
        if foot_active and (part_json["foot"] is not None):
            new_image.paste(part_imgs["foot"][0], self.get_coords(part_json['foot'])[0])  # 원하는 위치에 붙임 
            new_image.paste(part_imgs["foot"][1], self.get_coords(part_json['foot'])[1])  # 원하는 위치에 붙임 
        # 다른 파트들에 대해서도 같은 방식으로 처리
        return new_image
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        print(img_path)
        label = 0 if (img_path.split('/')[-1].split('.')[0].split('-')[0])=='A' else 1
        image = Image.open(img_path)
        part_name = ["head", "eye", "nose", "ear", "mouth", "hand", "foot", "upper_body", "lower_body"]
        if self.binary_thresholding:
            image = image.convert("L")#! Convert grayscale
            image = image.point(lambda p: p > self.binary_thresholding and 255)
        part_json = self.get_part_json(os.path.join(self.json_folder,self.json_paths[idx]),part_name=part_name)
        part_imgs = {}
        for part in part_name:#모든 part를 다시 dict으로 리턴하기위함.
            part_imgs[part]=[]
            # print(part)
            coords = self.get_coords(part_json[part])
            # print(coords)
            if coords is None:
                part_imgs[part].append(None)    
                
            elif len(coords) ==1:
                part_imgs[part].append(image.crop(coords[0]))    
            elif len(coords) == 2:
                part_imgs[part].append(image.crop(coords[0]))    
                part_imgs[part].append(image.crop(coords[1]))    
        empty_face = self.get_empty_face(image,part_imgs,part_json)
        # empty_face.show()
        empty_upper_body = self.get_empty_upper_body(image,part_imgs,part_json)
        empty_lower_body = self.get_empty_lower_body(image,part_imgs,part_json)
        part_imgs['empty_face']=[empty_face]
        part_imgs['empty_lower_body']=[empty_lower_body]
        part_imgs['empty_upper_body']=[empty_upper_body]
        part_combinations = list(itertools.product([0, 1], repeat=7))
        new_imgs = []
        for combination in part_combinations:
            # print(combination)
            new_img=self.create_new_images(img=image,binary_combination=combination, part_imgs=part_imgs,part_json=part_json)
            if self.transform:
                new_img=self.transform(new_img)
            new_imgs.append(new_img.unsqueeze(0))
        new_imgs = torch.cat(new_imgs,dim=0)
        return new_imgs,label 
    
    


if __name__=="__main__":
    transform= transforms.Compose([transforms.Resize((224,168)),transforms.ToTensor()])
    dataset = shapely_part('/data/jong980812/project/mae/util','/data/jong980812/project/mae/util',240,transform=transform)
    data_loader=DataLoader(dataset,2,num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for new_imgs,label in data_loader:
        print(new_imgs.shape)
        input_data = new_imgs
        print('complete')
        batch_size = input_data.shape[0]
        input_data = rearrange(input_data,  'b t c h w -> (b t) c h w')
        
        model.to(device)
        input_data = input_data.to(device)
        label = label.to(device)
        model.eval()
        with torch.no_grad():
            output=model(input_data)
        output = rearrange(output, '(b t) o -> b t o', b=batch_size) # batch_size, 128, output(2)
        print(output.shape)
        print(label)
