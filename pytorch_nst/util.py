import torch
import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from pytorch_nst.config import device, imsize

loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()
])
unloader = transforms.ToPILImage()

plt.ion()

def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0) # wrap image in tensor
    return image.to(device, torch.float)

def tensor_to_image(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    return unloader(image)

def imshow(tensor, title=None):
    image = tensor_to_image(tensor)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(1)

def show_all_images(content, style, generated, title=None):
    c_image = tensor_to_image(content)
    s_image = tensor_to_image(style)
    g_image = tensor_to_image(generated)
    
    fig = plt.figure(figsize=(6,6))
    grid = plt.GridSpec(3,2)

    c_ax = fig.add_subplot(grid[-1,0])
    c_ax.axis('off')
    c_ax.imshow(c_image)
    c_ax.title.set_text("Content Image")

    s_ax = fig.add_subplot(grid[-1,1])
    s_ax.axis('off')
    s_ax.imshow(s_image)
    s_ax.title.set_text("Style Image")
    
    g_ax = fig.add_subplot(grid[0:2, :])
    g_ax.axis('off')
    g_ax.imshow(g_image)

    if title is not None:
        plt.title(title)

def save_image(tensor, filepath):
    image = tensor_to_image(tensor)
    image.save(filepath)

def random_img():
    return torch.rand(1, 3, imsize, imsize).to(device, torch.float)
