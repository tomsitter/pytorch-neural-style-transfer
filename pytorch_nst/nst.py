import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy

from pytorch_nst.config import device, content_layers_default, style_layers_default

class ContentLoss(nn.Module):
    ''' Calculate mse loss between two feature maps from the same layer'''
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # must detach target from the tree so that
        # gradient can be computed dynamically
        self.target = target.detach()

    def forward(self, input):
        # Calculate mse loss and return the input. 
        # Since we return the layer unaltered, this layer has no actual impact on subsequent layers'''
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    ''' A gram matrix will calculate the relation between the filter activated caused by an image.
        Multiple filters being activated together by in image can be thought of as it's stlye
    '''
    batch_size, num_features, height, width = input.size()
    features = input.view(batch_size*num_features, height*width)
    G = torch.mm(features, features.t()) # gram product
    # normalize by dividing by number of elements. 
    # This is needed so that layers with different dimensions have the same weight
    return G.div(batch_size*num_features*height*width)

class StyleLoss(nn.Module):
    ''' Calculate mse loss of the gram matrix of the two feature maps from the same layer'''
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    ''' The VGG19 network was pre-trained with normalized images, so we will apply the same normalization
        to images fed into this application
    '''
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        
        self.mean = mean.clone().view(-1, 1, 1)
        self.std = std.clone().view(-1, 1, 1)

    def forward(self, img):
        #normalize image
        return (img - self.mean) / self.std

def get_style_model_and_losses(cnn, 
                           normalization_mean, normalization_std,
                           style_img, content_img, 
                           content_layers=content_layers_default,
                           style_layers=style_layers_default):
    '''
    We're going to build a model where we insert our content/style layers into the pre-trained cnn
    These layers are "transparent", as they pass along the input unaltered, but have their own custom
    loss functions that our optimizer will use on the generated image.
    '''
    cnn = copy.deepcopy(cnn)

    normalization = Normalization(normalization_mean, 
                                  normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i+=1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f'content_loss_{i}', content_loss)
            content_losses.append(content_loss)
        
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f'style_loss_{i}', style_loss)
            style_losses.append(style_loss)

    # trim layers after the last content/style loss layer, since we don't need them
    for i in range(len(model) -1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i+1)]

    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(cnn, 
                       normalization_mean, normalization_std,
                       content_img, style_img, input_img, 
                       num_steps=300, 
                       style_layers=None, style_weight=1000000, content_weight=1):

    if not style_layers:
        style_layers = style_layers_default

    print("Building the style transfer model..")
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                normalization_mean, normalization_std, 
                style_img, content_img, 
                style_layers=style_layers)
    optimizer = get_input_optimizer(input_img)

    print("Optimizing...")
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # clip pixel values to between 0 and 1
            input_img.data.clamp_(0,1)

            # Run a step forward, calculate loss of our content/style layers
            optimizer.zero_grad()
            model(input_img)
            style_score=0
            content_score=0

            for s1 in style_losses:
                style_score += s1.loss
            for c1 in content_losses:
                content_score += c1.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f'run {run}')
                print(f'Style Loss: {round(style_score.item(), 2)}  '
                      f'Content Loss: {round(content_score.item(), 2)}')
                print()
            
            return style_score + content_score
        
        optimizer.step(closure)

    # clip pixel values to between 0 and 1
    input_img.data.clamp_(0, 1)

    return input_img