import argparse
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T


from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
import os
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image,preprocess_image_no_normarlize
from pytorch_grad_cam.ablation_layer import AblationLayerVit
import models_vit
from util.pos_embed import interpolate_pos_embed

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image_path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')
    
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--finetune', default=None,
                        help='finetune from checkpoint')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir',default='',type=str)
    
    parser.add_argument('--target_classes', default=0, type=int,
                        help='본인이 알고싶은 클래스 gradcam')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python vit_gradcam.py --image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.

    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")
    if  'original_vit' in args.model:
            model= models_vit.__dict__[args.model](
            pretrained=True if args.finetune is None else False,
            num_classes=args.nb_classes,
            drop_path_rate=0.1,
            )
    elif 'resnet' in args.model:
        model= models_vit.__dict__[args.model](
        num_classes=args.nb_classes
        )
    else:
        model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        )
    # model = torch.hub.load('facebookresearch/deit:main',
    #                        'deit_tiny_patch16_224', pretrained=True)
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        # for k in ['head.weight', 'head.bias']:
        #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        #         print(f"Removing key {k} from pretrained checkpoint")
        #         del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    model.eval()

    if args.use_cuda:
        model = model.cuda()

    target_layers = [model.blocks[-1].norm1]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    if args.method == "ablationcam":
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform)

    rgb_img = Image.open(args.image_path)
    transform_resize = T.Resize((224,224))
    rgb_img = transform_resize(rgb_img)
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image_no_normarlize(rgb_img)#! normalize 안하려고 일부로 만들긴함.

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    targets = args.target_classes if args.target_classes != -1 else None
    #! -1주면 자동으로 none들어가고 아니면 내가 지정한게 들어가게함.

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

#? model_predict는 모델logit의 argmax
    grayscale_cam,model_predict = cam(input_tensor=input_tensor,
                        targets=targets,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)
    if model_predict==0:
        class_predicted='ASD'
    elif model_predict==1:
        class_predicted='TD'
    else:
        pass
    #! 나중에 추가될수도있으니 일단 하드코딩
    
    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    
    #! 이 경우 내가 target을 지정하지 않았으므로 모델이 뭘 찍었는지 알려줘야함.
    if targets is None:
        print(f'model predict {class_predicted}')
        
    
    img_name= \
    (args.image_path.split('/')[-1]).split('.')[0] +f'_{class_predicted}_'+'Model predict.png'if args.target_classes==-1 \
    else (args.image_path.split('/')[-1]).split('.')[0] +f'_{class_predicted}'+'.png'
    cv2.imwrite(os.path.join(args.output_dir,img_name), cam_image)