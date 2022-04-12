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


i0 = transforms.ToTensor()(Image.open('demo_imgs/4.png').convert('RGB'))
i1 = transforms.ToTensor()(Image.open('demo_imgs/6.png').convert('RGB'))
i2 = transforms.ToTensor()(Image.open('demo_imgs/8.png').convert('RGB'))
i3 = transforms.ToTensor()(Image.open('demo_imgs/11.png').convert('RGB'))


sz = 112*8
out_img = np.zeros((3,2*sz,2*sz))

out_img[:, 0:sz, 0:sz] = i0[:, 0:sz, 0:sz]
out_img[:, 0:sz, sz:sz*2] = i1[:, 0:sz, 0:sz]
out_img[:, sz:sz*2, 0:sz] = i2[:, 0:sz, 0:sz]
out_img[:, sz:sz*2, sz:sz*2] = i3[:, 0:sz, 0:sz]

plt.imshow(out_img.transpose(1,2,0))
plt.show()
im = Image.fromarray((out_img.transpose(1,2,0)* 255).astype(np.uint8))
im.save("demo_imgs/splice.png")