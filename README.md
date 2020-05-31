# PyTorch Neural Style Transfer
Command line application to run a neural style transfer using PyTorch based on this [PyTorch tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html). 

Content and Style images must be the same size. I have been resizing to 512x512. If you do not have a compatible GPU, the images will be automatically resized to 128x128.

### Install

On Linux, you may need to install python3-tk for image visualization

Download or clone this repository

Recommended: Create a virtualenv and activate it

Run `pip install -r requirements.txt`

### Usage

```
Usage: python main.py [OPTIONS]

Options:
  -c, --content TEXT      path to content image  [default: ./examples/content.jpg]

  -s, --style TEXT        path to style image  [default: ./examples/style.jpg]
  -o, --output TEXT       path to save generated image  [default: ./output/generated-1590787599.jpg]

  --style_weight INTEGER  [default: 1000000]
  --style_layers TEXT     Conv Layers to use for style loss  [default: 1,2,3,4,5]

  --steps INTEGER         [default: 300]
  --random_input          Will start with random noise if set, otherwise
                          content image

  --help                  Show this message and exit.
```

### Example:
  
`python main.py --content='./examples/bootsy.jpg' --style='./examples/picasso.jpg' --style_weight=500000 --steps=3000 --random_input`

![Image of Picasso Bootsy](https://github.com/tomsitter/pytorch-neural-style-transfer/blob/master/output/bootsy_picasso_3000_steps.png)
  
  
### Other Example:
![Image of Muted Picasso Booty](https://github.com/tomsitter/pytorch-neural-style-transfer/blob/master/output/bootsy_final.jpg)
  
### To Do:
* Add more command-line options  (e.g. which conv layers to use)
* Experiment with a ResNet-50 instead of VGG-19
* Package with setuptools
