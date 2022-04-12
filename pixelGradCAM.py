import argparse
from types import SimpleNamespace

from yaml import load
from os.path import join
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from scipy.ndimage.filters import gaussian_filter
import torch
from torch import nn
from GradCAM import GradCAM
from util import image_to_grid, load_weights, set_requires_grad, create_model
from scipy.ndimage.filters import gaussian_filter
from PIL import ImageFilter

mean = torch.Tensor([0.7806, 0.7806, 0.7806])
std = torch.tensor([0.0089, 0.0089, 0.0089])

def pixelGradCAM(model, img, layer, idx_to_class, plot_intermediate=False):
    img_height = img.shape[1]
    img_width = img.shape[2]

    grad_cam = GradCAM(model, layer, img_width, img_height, mean, std)
    target_class = list(range(len(idx_to_class.keys())))

    stack = []
    if plot_intermediate:
        plot_dims = int(np.ceil(np.sqrt(len(idx_to_class.keys()))))
        fig, ax = plt.subplots(plot_dims-1, plot_dims)
        ax = ax.ravel()
    
    for t in target_class:
        img_out, cam = grad_cam.generate_cam(img, t, normalize=False)
        stack.append(cam)

        if plot_intermediate:
            ax[t].imshow(img_out)
            ax[t].imshow(cam, cmap='jet', alpha=0.5)
            ax[t].axis('off')
            ax[t].set_title(f'Map for {idx_to_class[t]}')
    
    cams = torch.stack(stack, dim=2)
    cam_argmax = cams.argmax(dim=2)
    return img, cam_argmax


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    idx_to_class = {
        0: '02.5', 1: '03.5', 2: '04.0', 3: '04.5', 
        4: '05.0', 5: '05.5', 6: '06.0', 7: '06.5', 
        8: '07.0', 9: '07.5', 10: '08.0', 11: '08.5', 
        12: '09.0', 13: '09.5', 14: '10.0', 15: '11.0', 
        16: '11.5', 17: '12.0', 18: '12.5', 19: '13.0'}

    # ----------------- Setup Model & Load weights -----------------
    img_path =  'demo_imgs/splice.png'

    with open(join('model', 'args.txt'), 'r') as f:
        cfg = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    
    model = create_model(cfg.backbone, num_classes=20, freeze_backbone=True, head_type=cfg.head_type)
    model.eval()

    load_weights(model, None, 'model/model.pt')
    set_requires_grad(model, True)
    model = model.cuda()

    original_image = Image.open(img_path).convert('RGB')

    layer = 7
    c, h , w, = transforms.ToTensor()(original_image).shape
    patch_sz = 224
    input_images_list = transforms.ToTensor()(original_image).squeeze().unfold(1, patch_sz, patch_sz). \
                            unfold(2, patch_sz, patch_sz). \
                            reshape(3, -1, patch_sz, patch_sz).permute(1, 0, 2, 3)

    images = []
    cam_argmaxes = []
    for image_as_patch in list(input_images_list):
        img, cam_argmax = pixelGradCAM(model, image_as_patch.cuda(), layer, idx_to_class)
        images.append(img.cpu().detach())
        cam_argmaxes.append(cam_argmax.unsqueeze(0).cpu().detach())
        
    # assemble the image from the individual patches again
    images_whole = image_to_grid(images, h//patch_sz, w//patch_sz)
    cam_argmax_whole = image_to_grid(cam_argmaxes, h//patch_sz, w//patch_sz).squeeze()


    idx_to_class_float = {k: float(v) for k,v in idx_to_class.items() }
    class_cam_argmax = np.vectorize(idx_to_class_float.get)(cam_argmax_whole)


    smooth_cam_argmax = gaussian_filter(class_cam_argmax, sigma=25)
    fig, ax = plt.subplots(1,1)
    ax.imshow(images_whole.permute(1,2,0))
    ax.axis('off')

    smooth_cam_argmax[0,0] = 13.0
    smooth_cam_argmax[0,1] = 2.5

    im = ax.imshow(smooth_cam_argmax, cmap='plasma', alpha=0.6)
    cax = fig.add_axes()    # fig.colorbar(ax)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')

    cbar.ax.set_ylabel('Grain Density' )

    fig.set_size_inches([3.51, 2.17])
    fig.subplots_adjust(top=0.95,
                        bottom=0.015,
                        left=0.0,
                        right=0.975)
    plt.show()
