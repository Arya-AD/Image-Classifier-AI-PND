# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 08:21:02 2023

@author: Ariane Djeupang
"""

import argparse 
from train_helper import load_data 
from train_helper import MyModel 
import os 
    
import torch 
from torch import nn 
from torch import optim

##  Define our parameters

parser = argparse.ArgumentParser(description='Train a neural network model to classify flowers')

parser.add_argument('data_directory', action="store")
parser.add_argument('--save_dir', action="store", dest="save_dir", default="." ) 
parser.add_argument('--arch', action="store", dest="arch", default="vgg16") 
parser.add_argument('--learning_rate', action="store",type=float, dest="learning_rate", default=0.001) 
parser.add_argument('--hidden_units', action="store",type=int, dest="hidden_units", default=1500)
parser.add_argument('--epochs', action="store",type=int, dest="epochs", default=4)
parser.add_argument('--gpu', action="store_true", dest="gpu", default=False)

args = parser.parse_args()


# Input data via command line 
data_directory = args.data_directory
save_dir = args.save_dir
arch = args.arch
learning_rate = args.learning_rate 
hidden_units = args.hidden_units 
epochs = args.epochs 
gpu  = args.gpu


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

    
trainloader, validloader, testloader, class_to_idx = load_data(data_directory)

## Create  new instance of our model
model = MyModel(arch, hidden_units)

##  Verify if cuda is available
message_cuda = "cuda is available" if torch.cuda.is_available() else "cuda is not available"
print(message_cuda)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.NLLLoss()

# Train the classifier parameters 
optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

# Move the model to cuda if available.
model.to(device);


##  Starting points
steps = 0
running_loss = 0
print_every = 20


for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate the accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()        
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                  f"Valid accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()                    

 # TODO: Do validation on the test set           
test_loss = 0 
accuracy = 0 

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
                    
        test_loss += batch_loss.item()
                    
        # Calculate the accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        
print(f"Test loss: {test_loss/len(testloader):.3f}.. "
      f"Test accuracy: {accuracy/len(testloader):.3f}")                              

# TODO: Save the checkpoint 

checkpoint = { 'arch' : arch ,
               'state_dict' : model.state_dict(),
               'class_to_idx' : class_to_idx, 
               'hidden_units': hidden_units               
             }

if save_dir.endswith('/'):
    torch.save(checkpoint, save_dir + 'checkpoint.pth')
else:
    torch.save(checkpoint, save_dir + '/' + 'checkpoint.pth')
    
print("Checkpoint successfully saved!")