import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 128

# VGG network was trained on normalized images, so must do the same to ours 
cnn_normalization_mean = torch.tensor([[0.485, 0.456, 0.406]]).to(device)
cnn_normalization_std = torch.tensor([[0.229, 0.224, 0.225]]).to(device)

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']