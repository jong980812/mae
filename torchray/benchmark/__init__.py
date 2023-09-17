r"""This script provides a few functions for getting and plotting example data.
"""
import os
import torchvision
from matplotlib import pyplot as plt

from .datasets import *  # noqa
from .models import *  # noqa
import torch

def get_example_data(arch='vgg16', shape=224, weight=None,img_path=None):
    """Get example data to demonstrate visualization techniques.

    Args:
        arch (str, optional): name of torchvision.models architecture.
            Default: ``'vgg16'``.
        shape (int or tuple of int, optional): shape to resize input image to.
            Default: ``224``.

    Returns:
        (:class:`torch.nn.Module`, :class:`torch.Tensor`, int, int): a tuple
        containing

            - a convolutional neural network model in evaluation mode.
            - a sample input tensor image.
            - the ImageNet category id of an object in the image.
            - the ImageNet category id of another object in the image.

    """

    # Get a network pre-trained on ImageNet.
    # weight='/data/jong980812/project/mae/result/eff_b1/IN_/OUT/07/checkpoint-19.pth'
    
    model = torchvision.models.efficientnet_b1(pretrained=True)
    model.classifier[1] = torch.nn.Linear(1280,2)
    checkpoint = torch.load(weight, map_location='cpu')

    print("Load pre-trained checkpoint from: %s" % weight)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    # Switch to eval mode to make the visualization deterministic.
    model.eval()

    # We do not need grads for the parameters.
    for param in model.parameters():
        param.requires_grad_(False)

    # Download an example image from wikimedia.
    import requests
    from io import BytesIO
    from PIL import Image

    # url = 'https://encrypted-tbn1.gstatic.com/licensed-image?q=tbn:ANd9GcREj22c-wMNL5IDmU99v8G7voUl17Yxm0JJqMLqttdPT4DnaB99zqVK7HWiNzjP3aZnzCEf-ikAqb2yiDk'
    # response = requests.get(url)
    img = Image.open(img_path)

    # Pre-process the image and convert into a tensor
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

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

        plt.subplot(batch_size, 2, 1 + 2 * i)
        imsc(input[i])
        plt.title('input image', fontsize=8)

        plt.subplot(batch_size, 2, 2 + 2 * i)
        imsc(saliency[i], interpolation='none')
        plt.title('{} for category {} ({})'.format(
            method, IMAGENET_CLASSES[class_i], class_i), fontsize=8)

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
