# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 08:30:47 2023

@author: Ariane Djeupang 
"""

import torch
from train_helper import MyModel 
from PIL import Image
from torchvision import transforms  


#   Preprocess the given image of flower we want to predict the class
def process_image(image):
    
    # Define the transformations properties
    transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    # Apply the transformations to the image
    processed_image = transform(image)

    # Convert the PyTorch tensor to a NumPy array
    numpy_image = processed_image.numpy()

    return numpy_image


#   Load the checkpont
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    model = MyModel(arch, hidden_units)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model 


#  Predict the class of a flower
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of a flower using a trained neural network model.
    '''
    
    # TODO: Predict the class from an image file
    im = process_image(Image.open(image_path))
    im = torch.from_numpy(im)
    im.unsqueeze_(0)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ## Load the model and the image to device
    im = im.to(device, dtype=torch.float)
    model.to(device); 
    
    ## Make the prediction using our model
    ps = torch.exp(model(im)) 
    
    ##Looking for top probability and top class
    top_p, top_class = ps.topk(topk, dim=1)
    
    ##Convert the output of top probabilities and top classes to list
    top_p = top_p.cpu().detach().numpy()[0]
    top_class = top_class.cpu().detach().numpy()[0]
    
    return top_p, top_class 
