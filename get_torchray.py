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


def get_example_data(arch='resnet18', shape=224, img_path=None, checkpoint=None):
    # Get a network pre-trained on ImageNet.
    model= models_vit.__dict__[arch](num_classes=2)
    
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

    # Pre-process the image and convert into a tensor
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((shape, shape)),
        torchvision.transforms.CenterCrop(shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.9761, 0.9771, 0.9794], std=[0.0797, 0.0790, 0.0766]),
    ])

    x = transform(img).unsqueeze(0)

    # bulldog category id.
    category_id_1 = 0

    # persian cat category id.
    category_id_2 = 1

    # Move model and input to device.
    from torchray.utils import get_device
    dev = get_device()
    model = model.to(dev)
    x = x.to(dev)

    return model, x, category_id_1, category_id_2
import cv2
def plot_example(input,
                 saliency,
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
    from torchray.utils import imsc
    from torchray.benchmark.datasets import IMAGENET_CLASSES

    if isinstance(category_id, int):
        category_id = [category_id]

    batch_size = len(input)

    plt.clf()
    for i in range(batch_size):
        class_i = category_id[i % len(category_id)]

        # imsc(input[i] * saliency[i], interpolation='none')
        img = (input[i] * saliency[i] * 3).detach().cpu().permute(1, 2, 0).numpy()
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
        # plt.imshow(img, cmap='gray')
        plt.imshow(img)
        plt.title('category ({})'.format(class_i))

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
        plt.savefig(save_path, format=ext, bbox_inches='tight')

    # Show plot if desired.
    if show_plot:
        plt.show()
        
####################################################################################

with open("ls-with-split.txt", 'r') as f :
    img_list = f.readlines()

for img_name in img_list :
    # Obtain example data.
    split = img_name.split('/')[0]
    checkpoint = torch.load(f"/data/ahngeo11/mae/result/resnet18/finetune/compact_2-eq1/resnet18/imgnet/20e/1e-3/OUT/{split}/checkpoint-19.pth", map_location='cpu')
    model, x, category_id_1, category_id_2 = get_example_data(img_path="/local_datasets/asd/compact_crop_trimmed_2/" + img_name.replace('\n', ''),
                                                              checkpoint=checkpoint)

    # Run on GPU if available.
    device = get_device()
    model.to(device)
    x = x.to(device)

    # Extremal perturbation backprop.
    masks_1, _ = extremal_perturbation(
        model, x, category_id_1,
        reward_func=contrastive_reward,
        debug=True,
        areas=[0.1],
    )

    masks_2, _ = extremal_perturbation(
        model, x, category_id_2,
        reward_func=contrastive_reward,
        debug=True,
        areas=[0.1],
    )

    # Plots.
    plot_example(x, masks_1, 'extremal perturbation', category_id_1, save_path=os.path.join("/local_datasets/asd/perturbation", img_name.split('/')[-1].split('.')[0] + "-about-ASD.jpg"))
    plot_example(x, masks_2, 'extremal perturbation', category_id_2, save_path=os.path.join("/local_datasets/asd/perturbation", img_name.split('/')[-1].split('.')[0] + "-about-TD.jpg"))
    
    # torch.save(masks_1, os.path.join("/local_datasets/asd/perturbation/mask", img_name.split('/')[-1].split('.')[0] + "-about-ASD.pth"))
    # torch.save(masks_2, os.path.join("/local_datasets/asd/perturbation/mask", img_name.split('/')[-1].split('.')[0] + "-about-TD.pth"))
    