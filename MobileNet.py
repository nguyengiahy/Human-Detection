!unzip images.zip

import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn   # All neural netword modules
import torch.optim as optim   # All optimizers
import torch.nn.functional as F # All functions that don't have any parameters
from torchvision import models, datasets, transforms  # For pretrained models, popular datasets, transformations to perform on images
from torch.utils.data import DataLoader   # To do dataset management
from collections import OrderedDict

# Data directories
train_dir = './train'
valid_dir = './valid'
test_dir = './test'

# Define transforms for training, validation, and test sets
training_transforms = transforms.Compose([
                                      transforms.Grayscale(num_output_channels=3),
                                      transforms.Resize(256),
                                      transforms.RandomRotation(15),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.5)),
                                      transforms.ColorJitter(brightness=(1, 1.5), hue=(-0.1, 0.1)),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
validation_transforms = transforms.Compose([
                                      transforms.Grayscale(num_output_channels=3),
                                      transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
testing_transforms = transforms.Compose([
                                     transforms.Grayscale(num_output_channels=3),
                                     transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
# Load the dataset
training_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)
validation_dataset = datasets.ImageFolder(valid_dir, transform=data_transforms)
testing_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)

# Define dataloader
train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=256, shuffle=True, num_workers=4)
validate_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=256, num_workers=4)
test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=128, num_workers=4)

# Model
model = models.mobilenet_v2(pretrained=True)

# Fine tune
for param in model.parameters():
  param.requires_grad = False  # Freeze the pretrained model parameters, avoid backpropagation to them

# Build custom classifier
classifier = nn.Sequential(OrderedDict([
                                        ('fc1', nn.Linear(in_features=1280, out_features=120)),
                                        ('relu', nn.ReLU(inplace=True)),
                                        ('drop', nn.Dropout(p=0.2, inplace=False)),
                                        ('fc2', nn.Linear(in_features=120, out_features=2)),
                                        ('output', nn.LogSoftmax(dim=1))]))
model.classifier = classifier

# Evaluate model
def validation(model, validateloader, criterion):

  val_loss = 0
  accuracy = 0

  for images, labels in iter(validateloader):
    images, labels = images.to('cuda'), labels.to('cuda')
    
    output = model.forward(images)
    val_loss += criterion(output, labels).item()

    probabilities = torch.exp(output)

    equality = (labels.data == probabilities.max(dim=1)[1])
    accuracy += equality.type(torch.FloatTensor).mean()
    
  return val_loss, accuracy

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# Train classifier layer
import copy
train_losses = []
val_losses = []
best_loss_model = copy.deepcopy(model)
best_acc_model = copy.deepcopy(model)

# Train classifier layer
def train_classifier():
  epochs = 100
  print_every = 8
  best_val_loss = int(1e9)
  best_acc = 0

  model.to('cuda')  # train model on GPU

  for e in range(epochs):

    model.train()   # in train mode
    running_loss = 0
    steps = 0

    for images, labels in iter(train_loader):   # take images by batches
      steps += 1
      images, labels = images.to('cuda'), labels.to('cuda')
      optimizer.zero_grad()   # zero the gradient: we want to recalculate them in every batches

      output = model.forward(images)
      loss = criterion(output, labels)
      loss.backward()
      optimizer.step()  # Run GD, update weights and biases

      running_loss += loss.item()

      if steps % print_every == 0:
        model.eval()
        
        # Turn off GD for validation, save resources 
        with torch.no_grad():
          validation_loss, accuracy = validation(model, validate_loader, criterion)

        print("Epoch: {}/{}..".format(e+1, epochs),
              "Training loss: {:.3f}..".format(running_loss/print_every),
              "Validation loss: {:.3f}..".format(validation_loss/len(validate_loader)),
              "Validation accuracy: {:.3f}".format(accuracy/len(validate_loader)))
        
        train_losses.append(running_loss/print_every)
        val_losses.append(validation_loss/len(validate_loader))

        # Save the best model
        if val_losses[-1] < best_val_loss:
          best_loss_model = copy.deepcopy(model)
          best_val_loss = val_losses[-1]
        if accuracy/len(validate_loader) > best_acc:
          best_acc = accuracy/len(validate_loader)
          best_acc_model = copy.deepcopy(model)

        running_loss = 0
        model.train()

  return best_val_loss, best_acc
best_val_loss, best_acc = train_classifier()

# Print plots
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(val_losses,label="val")
plt.plot(train_losses,label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Save model
model.save("model.h5")
best_loss_model.save("best_loss.h5")
best_acc_model.save("best_acc.h5")

print("best validation loss:", best_val_loss)
print("best validation acc:", best_acc)

# Load model
#best_acc_model = torch.load("best_acc.h5")
#best_loss_model = torch.load("best_loss.h5")

# Testing network
def test_accuracy(model, test_loader):

  model.eval()
  model.to('cuda')

  with torch.no_grad():
    accuracy = 0
    for images, labels in iter(test_loader):
      images, labels = images.to('cuda'), labels.to('cuda')

      output = model.forward(images)
      probabilities = torch.exp(output)

      equality = (labels.data == probabilities.max(dim=1)[1])
      accuracy += equality.type(torch.FloatTensor).mean()
    
    print('Test accuracy: {}'.format(accuracy/len(test_loader)))

test_accuracy(best_acc_model, test_loader)
test_accuracy(best_loss_model, test_loader)

# Precision and recall calculation
def performance(model, test_loader):

  model.eval()
  model.to('cuda')

  with torch.no_grad():
    tp = fp = tn = fn = 0
    for images, labels in iter(test_loader):
      images, labels = images.to('cuda'), labels.to('cuda')
      
      output = model.forward(images)
      probabilities = torch.exp(output)

      if labels == 1:
        if probabilities.max(dim=1)[1] == 1:
          tp += 1
        else:
          fn += 1
      
      if labels == 0:
        if probabilities.max(dim=1)[1] == 0:
          tn += 1
        else:
          fp += 1
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    return precision, recall, tp, fp, tn, fn

# Calculate precision, recall of best accuracy model
test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=1, num_workers=4)
precision, recall, tp, fp, tn, fn = performance(best_acc_model, test_loader)
print("precision:", precision)
print("recall", recall)
print("TP:", tp)
print("FP:", fp)
print("TN:", tn)
print("FN:", fn)

# Calculate precision, recall of best loss model
precision, recall, tp, fp, tn, fn = performance(best_loss_model, test_loader)
print("precision:", precision)
print("recall", recall)
print("TP:", tp)
print("FP:", fp)
print("TN:", tn)
print("FN:", fn)


# Pre-process input image
from PIL import Image 
def process_image(image_path):
  pil_image = Image.open(image_path)

  # Resize
  if pil_image.size[0] > pil_image.size[1]:
    pil_image.thumbnail((5000, 256))
  else:
    pil_image.thumbnail((256, 5000))
  
  # Crop
  left_margin = (pil_image.width-224)/2
  bottom_margin = (pil_image.height-224)/2
  right_margin = left_margin + 224
  top_margin = bottom_margin + 224

  # Normalize 
  np_image = np.array(pil_image)/255
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  np_image = (np_image - mean)/std

  # Pytorch expects color channel to be the first dimension but it's the third dimension in the PIL image and numpy array
  # Color channel needs to be first; retain the order of the other 2 dimensions 
  np_image = np_image.transpose([2,0,1])

  return np_image

# Predict an image
def predict(image_path, model, topk=1):
  image = process_image(image_path)
  # Convert image to pytorch tensor first
  image = torch.from_numpy(image).type(torch.cuda.FloatTensor)

  # Return new tensor with a dimension of size one inserted at specified position
  image = image.unsqueeze(0)
  output = model.forward(image)
  probabilities = torch.exp(output)

  # Highest probability and index of it
  best_probability, best_index = probabilities.topk(topk)

  # Convert to list
  best_probability = best_probability.detach().type(torch.FloatTensor).numpy().tolist()[0]
  best_index = best_index.detach().type(torch.FloatTensor).numpy().tolist()[0]

  # Convert topk_indices to actual class labels using class_to_idx
  # Invert the dictionary so you get a mapping from index to class
  idx_to_class = {value: key for key, value in model.class_to_idx.items()}
  top_classes = [idx_to_class[index] for index in best_index]

  return best_probability, top_classes