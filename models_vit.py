# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from torchvision import models
import torch
import torch.nn as nn

import timm.models.vision_transformer
import timm

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
def original_vit_base_patch16_224(pretrained,num_classes,**kwargs):
    model = timm.create_model('vit_base_patch16_224',pretrained=pretrained,num_classes=num_classes,**kwargs)
    return model
def original_vit_tiny_patch16_224(pretrained,num_classes,**kwargs):
    model = timm.create_model('vit_tiny_patch16_224',pretrained=pretrained,num_classes=num_classes,**kwargs)
    return model
def original_vit_small_patch16_224(pretrained,num_classes,**kwargs):
    model = timm.create_model('vit_small_patch16_224',pretrained=pretrained,num_classes=num_classes,**kwargs)
    return model
def original_vit_tiny_patch16_224_in21k(**kwargs):
    model = timm.create_model('vit_tiny_patch16_224_in21k',pretrained=True,num_classes=2,**kwargs)
    return model

def resnet18(num_classes,pretrained):
    model = models.resnet18(pretrained=pretrained)
    in_ft = model.fc.in_features #모델의 마지막 fc layer in feature
    model.fc = nn.Linear(in_ft,num_classes)
    return model


'''
'vit_base_patch16_224', 
'vit_base_patch16_224_in21k',
'vit_base_patch16_224_miil',
'vit_base_patch16_224_miil_in21k', 
'vit_base_patch16_384',
'vit_base_patch32_224', 
'vit_base_patch32_224_in21k',
'vit_base_patch32_384',
'vit_base_r50_s16_224_in21k', 
'vit_base_r50_s16_384',
'vit_huge_patch14_224_in21k', 
'vit_large_patch16_224',
'vit_large_patch16_224_in21k', 
'vit_large_patch16_384', 
'vit_large_patch32_224_in21k',
'vit_large_patch32_384',
'vit_large_r50_s32_224',
'vit_large_r50_s32_224_in21k', 'vit_large_r50_s32_384',
'vit_small_patch16_224', 'vit_small_patch16_224_in21k', 
'vit_small_patch16_384', 'vit_small_patch32_224', 
'vit_small_patch32_224_in21k', 'vit_small_patch32_384', 
'vit_small_r26_s32_224', 'vit_small_r26_s32_224_in21k',
'vit_small_r26_s32_384', 'vit_tiny_patch16_224',
'vit_tiny_patch16_224_in21k', 'vit_tiny_patch16_384', 
'vit_tiny_r_s16_p8_224', 'vit_tiny_r_s16_p8_224_in21k', 
'vit_tiny_r_s16_p8_384', 'wide_resnet50_2', 'wide_resnet101_2', 
'xception', 'xception41', 'xception65', 'xception71']
'''