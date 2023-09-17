# from torchray.attribution.grad_cam import grad_cam
# from torchray.benchmark import plot_example

import os
import torchvision
from matplotlib import pyplot as plt
import torch
import models_vit

from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward
from torchray.utils import get_device

# from .datasets import *  # noqa
# from .models import *  # noqa
from torchvision import models
from util.datasets import AI_HUB
import argparse
import cv2


def get_example_data(arch='resnet18', shape=224, img_path=None, checkpoint=None):
    # Get a network pre-trained on ImageNet.
    # model= models_vit.__dict__[arch](num_classes=2)
    
    model=models.efficientnet_b1(pretrained=False)
    model.classifier[1] = torch.nn.Linear(1280, 2)
    
    checkpoint_model = checkpoint['model']

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    # Switch to eval mode to make the visualization deterministic.
    model.eval()

    # We do not need grads for the parameters.
    for param in model.parameters():
        param.requires_grad_(False)

    # Download an example image from wikimedia.
    from PIL import Image

    # url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/Arthur_Heyer_-_Dog_and_Cats.jpg/592px-Arthur_Heyer_-_Dog_and_Cats.jpg'
    # response = requests.get(url)
    img = Image.open(img_path)

        # torchvision.transforms.CenterCrop(shape),
    # Pre-process the image and convert into a tensor
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean=[0.96, 0.96, 0.96], std=[0.1, 0.1, 0.1]),
    ])
    
    no_norm_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])

    x = transform(img).unsqueeze(0)
    original_x = no_norm_transform(img).unsqueeze(0)
    

    # bulldog category id.
    category_id_1 = 0

    # persian cat category id.
    category_id_2 = 1

    # Move model and input to device.
    from torchray.utils import get_device
    dev = get_device()
    model = model.to(dev)
    x = x.to(dev)

    return model, x, category_id_1, category_id_2, original_x


def plot_example(input,
                 saliency,
                 original_input,
                 method,
                 category_id,
                 show_plot=False,
                 save_path=None):
    """Plot an example.

    Args:
        input (:class:`torch.Tensor`): 4D tensor containing input images.
        saliency (:class:`torch.Tensor`): 4D tensor containing saliency maps.
        method (str): name of saliency method.
        category_id (int): ID of ImageNet category.
        show_plot (bool, optional): If True, show plot. Default: ``False``.
        save_path (str, optional): Path to save figure to. Default: ``None``.
    """
    # from torchray.utils import imsc
    topil = torchvision.transforms.ToPILImage()

    if isinstance(category_id, int):
        category_id = [category_id]

    batch_size = len(input)

    plt.clf()
    for i in range(batch_size):
        class_i = category_id[i % len(category_id)]

        # imsc(input[i] * saliency[i], interpolation='none')
        mask = torch.clamp(saliency[i], 0.4, 1.0)
        img = topil((original_input[i] * mask.detach().cpu()))
        
        #* to plt.show
        # img = (input[i] * mask).detach().cpu().permute(1, 2, 0).numpy()
        # plt.imshow(img)
        # plt.title('category ({})'.format(class_i))

        # plt.subplot(batch_size, 2, 1 + 2 * i)
        # imsc(input[i])
        # plt.title('input image', fontsize=8)

        # plt.subplot(batch_size, 2, 2 + 2 * i)
        # imsc(saliency[i], interpolation='none')
        # plt.title('{} for category {} ({})'.format(
        #     method, IMAGENET_CLASSES[class_i], class_i), fontsize=8)

    # Save figure if path is specified.
    if save_path:
        save_dir = os.path.dirname(os.path.abspath(save_path))
        # Create directory if necessary.
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ext = os.path.splitext(save_path)[1].strip('.')
        # plt.savefig(save_path, format=ext, bbox_inches='tight')
        img.save(save_path)

    # Show plot if desired.
    if show_plot:
        plt.show()
        
####################################################################################

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/data/dataset/asd/All_5split', type=str)
    parser.add_argument('--job_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    return parser.parse_args()

def main(args) :
    import os
    import json
    import numpy as np
    from tqdm import tqdm

    img_list = None
    root = args.data_root
    if not os.path.exists(root) :
        root = '/data/datasets/asd/All_5split'
        
    for i in range(1, 6):
        split_name = f"0{i}"
        for class_name in ['ASD', 'TD'] :
            split_path = os.path.join(root, split_name, 'val', class_name)
            if img_list is None :
                img_list = [os.path.join(split_path, elem) for elem in os.listdir(split_path)]
            else :
                img_list = img_list + [os.path.join(split_path, elem) for elem in os.listdir(split_path)]


    job_dir = args.job_dir

    for img_name in tqdm(img_list) :
        # Obtain example data.
        split_name = img_name.split('/')[-4]
        
        with open(os.path.join(job_dir, split_name + "_log.txt"), 'r') as f :
            data = f.readlines()
        
        accs = np.array([json.loads(line.replace('\n', ''))['test_acc1'] for line in data])
        best_and_last_epoch = np.max(np.where(accs == np.max(accs)))
        
        path = os.path.join(job_dir, split_name, f"checkpoint-{best_and_last_epoch}.pth")
        checkpoint = torch.load(path, map_location='cpu')
        
        model, x, category_id_1, category_id_2, original_x = get_example_data(img_path=img_name, checkpoint=checkpoint)

        # Run on GPU if available.
        device = get_device()
        model.to(device)
        x = x.to(device)

        # Extremal perturbation backprop.
        masks_1, _ = extremal_perturbation(
            model, x, category_id_1,
            reward_func=contrastive_reward,
            debug=True,
            areas=[0.05],
        )

        masks_2, _ = extremal_perturbation(
            model, x, category_id_2,
            reward_func=contrastive_reward,
            debug=True,
            areas=[0.05],
        )

        # Plots.
        
        save_path = args.save_dir
        
        plot_example(x, masks_1, original_x, 'extremal perturbation', category_id_1, save_path=os.path.join(save_path, "img", img_name.split('/')[-1].split('.')[0] + "-about-ASD.jpg"))
        plot_example(x, masks_2, original_x, 'extremal perturbation', category_id_2, save_path=os.path.join(save_path, "img", img_name.split('/')[-1].split('.')[0] + "-about-TD.jpg"))
        
        torch.save(masks_1, os.path.join(save_path, "mask", img_name.split('/')[-1].split('.')[0] + "-about-ASD.pth"))
        torch.save(masks_2, os.path.join(save_path, "mask", img_name.split('/')[-1].split('.')[0] + "-about-TD.pth"))


if __name__ == '__main__':
    args = get_args()
    main(args)