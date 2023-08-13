"""
Created on Thu Oct 26 14:19:44 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np
import copy
import torch
from torch.optim import SGD
from torchvision import models
from misc_functions import preprocess_image, recreate_image, save_image
from timm.models.layers import trunc_normal_
from torchvision import datasets, transforms
from PIL import Image

class ClassSpecificImageGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent
    """
    def __init__(self, model, target_class):
        # self.mean = [-0.485, -0.456, -0.406]
        # self.std = [1/0.229, 1/0.224, 1/0.225]
        self.mean = [-0.96, -0.96, -0.96]
        self.std = [1/0.1, 1/0.1, 1/0.1]
        self.model = model
        self.model.eval()
        self.target_class = target_class
        # Generate a random image
        self.created_image = np.uint8(np.random.uniform(140, 160, (224, 224, 3)))
        # self.created_image=np.uint8(Image.open('/data/datasets/tu_berlin/01/train/bicycle/1700.png'))#pillow객체)
        # self.created_image = np.repeat(self.created_image[:,:,np.newaxis],3,-1) 
        # Create the folder to export images if not exists
        if not os.path.exists('../generated/class_'+str(self.target_class)):
            os.makedirs('../generated/class_'+str(self.target_class))

    def generate(self, iterations=500):
        """Generates class specific image

        Keyword Arguments:
            iterations {int} -- Total iterations for gradient ascent (default: {150})

        Returns:
            np.ndarray -- Final maximally activated class image
            
        """
        topil=transforms.ToPILImage()
        initial_learning_rate = 1
        for i in range(1, iterations):
            # Process image and return variable
            self.processed_image = preprocess_image(self.created_image, True)
            # Define optimizer for the image
            optimizer = SGD([self.processed_image], lr=initial_learning_rate)
            # Forward
            output = self.model(self.processed_image)
            # Target specific class
            class_loss = -output[0, self.target_class]

            if i % 10 == 0 or i == iterations-1:
                print('Iteration:', str(i), 'Loss',
                      "{0:.2f}".format(class_loss.data.numpy()))
            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()
            self.created_image = recreate_image(self.processed_image)
            if i % 10 == 0 or i == iterations-1:
                self.save_img=copy.copy(self.processed_image.data.numpy()[0])
                reverse_mean = [-0.96, -0.96, -0.96]
                reverse_std = [1/0.1, 1/0.1, 1/0.1]
                for c in range(3):
                    self.save_img[c] /= reverse_std[c]
                    self.save_img[c] -= reverse_mean[c]
                # self.save_img[self.save_img > 0.5] = 255
                # self.save_img[self.save_img < 0.5] = 0
                self.save_img=(self.save_img.transpose(1,2,0))
                self.save_img =np.uint8(self.save_img * 255)
                topil=transforms.ToPILImage()
                self.save_img=topil(self.save_img)
                transform=transforms.Compose([
                transforms.Grayscale(3),
                transforms.ToTensor(),
                ])
                self.save_img=transform(self.save_img)
                # self.save_img[self.save_img > 0.3] = 1
                # self.save_img[self.save_img < 0.3] = 0
                self.save_img=topil(self.save_img)
                # Save image
                im_path = os.path.join(f'/data/jong980812/project/mae/pytorch-cnn-visualizations-master/generated/class_{self.target_class}','iter_'+str(i)+'.png')
                
                save_image(self.save_img, im_path)

        return self.processed_image


if __name__ == '__main__':
    target_class =1  # Flamingo
    # pretrained_model = models.alexnet(pretrained=True)
    model=models.efficientnet_b1(pretrained=True)
    model.classifier[1] = torch.nn.Linear(1280, 2)
    
    model_path='/data/jong980812/project/mae/result_ver2/All_5split/bs4_1e-2/OUT/01/checkpoint-29.pth'
    checkpoint = torch.load(model_path, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % model_path)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    csig = ClassSpecificImageGeneration(model, target_class)
    os.makedirs(f'/data/jong980812/project/mae/pytorch-cnn-visualizations-master/generated/class_{target_class}',exist_ok=True)
    csig.generate()



###TU berlin class
# {'airplane': 0, 'alarm clock': 1, 'angel': 2, 'ant': 3, 'apple': 4, 'arm': 5, 'armchair': 6, 'ashtray': 7, 'axe': 8, 'backpack': 9, 'banana': 10, 'barn': 11, 'baseball bat': 12, 'basket': 13, 'bathtub': 14, 'bear (animal)': 15, 'bed': 16, 'bee': 17, 'beer-mug': 18, 'bell': 19, 'bench': 20, 'bicycle': 21, 'binoculars': 22, 'blimp': 23, 'book': 24, 'bookshelf': 25, 'boomerang': 26, 'bottle opener': 27, 'bowl': 28, 'brain': 29, 'bread': 30, 'bridge': 31, 'bulldozer': 32, 'bus': 33, 'bush': 34, 'butterfly': 35, 'cabinet': 36, 'cactus': 37, 'cake': 38, 'calculator': 39, 'camel': 40, 'camera': 41, 'candle': 42, 'cannon': 43, 'canoe': 44, 'car (sedan)': 45, 'carrot': 46, 'castle': 47, 'cat': 48, 'cell phone': 49, 'chair': 50, 'chandelier': 51, 'church': 52, 'cigarette': 53, 'cloud': 54, 'comb': 55, 'computer monitor': 56, 'computer-mouse': 57, 'couch': 58, 'cow': 59, 'crab': 60, 'crane (machine)': 61, 'crocodile': 62, 'crown': 63, 'cup': 64, 'diamond': 65, 'dog': 66, 'dolphin': 67, 'donut': 68, 'door': 69, 'door handle': 70, 'dragon': 71, 'duck': 72, 'ear': 73, 'elephant': 74, 'envelope': 75, 'eye': 76, 'eyeglasses': 77, 'face': 78, 'fan': 79, 'feather': 80, 'fire hydrant': 81, 'fish': 82, 'flashlight': 83, 'floor lamp': 84, 'flower with stem': 85, 'flying bird': 86, 'flying saucer': 87, 'foot': 88, 'fork': 89, 'frog': 90, 'frying-pan': 91, 'giraffe': 92, 'grapes': 93, 'grenade': 94, 'guitar': 95, 'hamburger': 96, 'hammer': 97, 'hand': 98, 'harp': 99, 'hat': 100, 'head': 101, 'head-phones': 102, 'hedgehog': 103, 'helicopter': 104, 'helmet': 105, 'horse': 106, 'hot air balloon': 107, 'hot-dog': 108, 'hourglass': 109, 'house': 110, 'human-skeleton': 111, 'ice-cream-cone': 112, 'ipod': 113, 'kangaroo': 114, 'key': 115, 'keyboard': 116, 'knife': 117, 'ladder': 118, 'laptop': 119, 'leaf': 120, 'lightbulb': 121, 'lighter': 122, 'lion': 123, 'lobster': 124, 'loudspeaker': 125, 'mailbox': 126, 'megaphone': 127, 'mermaid': 128, 'microphone': 129, 'microscope': 130, 'monkey': 131, 'moon': 132, 'mosquito': 133, 'motorbike': 134, 'mouse (animal)': 135, 'mouth': 136, 'mug': 137, 'mushroom': 138, 'nose': 139, 'octopus': 140, 'owl': 141, 'palm tree': 142, 'panda': 143, 'paper clip': 144, 'parachute': 145, 'parking meter': 146, 'parrot': 147, 'pear': 148, 'pen': 149, 'penguin': 150, 'person sitting': 151, 'person walking': 152, 'piano': 153, 'pickup truck': 154, 'pig': 155, 'pigeon': 156, 'pineapple': 157, 'pipe (for smoking)': 158, 'pizza': 159, 'potted plant': 160, 'power outlet': 161, 'present': 162, 'pretzel': 163, 'pumpkin': 164, 'purse': 165, 'rabbit': 166, 'race car': 167, 'radio': 168, 'rainbow': 169, 'revolver': 170, 'rifle': 171, 'rollerblades': 172, 'rooster': 173, 'sailboat': 174, 'santa claus': 175, 'satellite': 176, 'satellite dish': 177, 'saxophone': 178, 'scissors': 179, 'scorpion': 180, 'screwdriver': 181, 'sea turtle': 182, 'seagull': 183, 'shark': 184, 'sheep': 185, 'ship': 186, 'shoe': 187, 'shovel': 188, 'skateboard': 189, 'skull': 190, 'skyscraper': 191, 'snail': 192, 'snake': 193, 'snowboard': 194, 'snowman': 195, 'socks': 196, 'space shuttle': 197, 'speed-boat': 198, 'spider': 199, 'sponge bob': 200, 'spoon': 201, 'squirrel': 202, 'standing bird': 203, 'stapler': 204, 'strawberry': 205, 'streetlight': 206, 'submarine': 207, 'suitcase': 208, 'sun': 209, 'suv': 210, 'swan': 211, 'sword': 212, 'syringe': 213, 't-shirt': 214, 'table': 215, 'tablelamp': 216, 'teacup': 217, 'teapot': 218, 'teddy-bear': 219, 'telephone': 220, 'tennis-racket': 221, 'tent': 222, 'tiger': 223, 'tire': 224, 'toilet': 225, 'tomato': 226, 'tooth': 227, 'toothbrush': 228, 'tractor': 229, 'traffic light': 230, 'train': 231, 'tree': 232, 'trombone': 233, 'trousers': 234, 'truck': 235, 'trumpet': 236, 'tv': 237, 'umbrella': 238, 'van': 239, 'vase': 240, 'violin': 241, 'walkie talkie': 242, 'wheel': 243, 'wheelbarrow': 244, 'windmill': 245, 'wine-bottle': 246, 'wineglass': 247, 'wrist-watch': 248, 'zebra': 249}