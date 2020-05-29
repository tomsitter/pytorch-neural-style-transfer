import os
import sys
import psutil
import logging
from time import time

import click
import matplotlib.pyplot as plt

import torchvision.models as models

from pytorch_nst.config import device, cnn_normalization_mean, cnn_normalization_std
from pytorch_nst.util import image_loader, imshow, random_img, save_image, show_all_images
from pytorch_nst.nst import run_style_transfer

def validate_layers(ctx, param, value):
    try:
        layers = [int(l) for l in value.split(',')]
        assert all(isinstance(l, int) for l in layers)
        return [f'conv_{l}' for l in layers]
    except ValueError:
        raise click.BadParameter('Layers need to be ints seperated by commas')

@click.command()
@click.option('-c', '--content', 
                default='./examples/content.jpg', 
                prompt='Content image',
                help='path to content image',
                show_default=True)
@click.option('-s', '--style', 
                default='./examples/style.jpg', 
                prompt='Style image',
                help='path to style image',
                show_default=True)
@click.option('-o', '--output', 
                default='./output/generated-' +  str(int(time())) + '.jpg', 
                prompt=False,
                help='path to save generated image',
                show_default=True)
@click.option('--style_weight', default=1000000, prompt=False, show_default=True)
@click.option('--style_layers', callback=validate_layers, 
                default='1,2,3,4,5', show_default=True,
                help='Conv Layers to use for style loss')
@click.option('--steps', default=300, prompt=False, show_default=True)
@click.option('--random_input', is_flag=True,
                help='Will start with random noise if set, otherwise content image')
def cli(content, style, output, style_weight, style_layers, steps, random_input):
    style_img = image_loader(style)
    content_img = image_loader(content)
    assert style_img.size() == content_img.size(), \
        "Style and Content images must be the same size"\

    if random_input:
        input_img = random_img()
    else:
        input_img = content_img.clone()

    # Load a pre-trained VGG network
    print("Loading pre-trained VGG19. This may take a while the first run...")
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    gen_img = run_style_transfer(cnn, 
                    cnn_normalization_mean, cnn_normalization_mean, 
                    content_img, style_img, input_img, 
                    style_layers=style_layers, style_weight=style_weight, 
                    num_steps=steps)

    show_all_images(content_img, style_img, gen_img, title="Generated Image")

    if click.confirm('Do you want to save the generated image?'):
        save_image(gen_img, output)
        print(f'Saved to {output}')

