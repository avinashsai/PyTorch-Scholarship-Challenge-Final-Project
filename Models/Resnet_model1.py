import re
import os
import sys
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torch.nn.functional as F
from torchvision import datasets, models, transforms


data_dir = 'flower_data/'
train_dir = data_dir+'train'
valid_dir = data_dir+'valid'


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'valid']}


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 102

model = torchvision.models.resnet152(pretrained=True)

print(model)


def pretrainedmodel():
  model = torchvision.models.resnet152(pretrained=True)
  
  for param in model.parameters():
    param.requires_grad = False
    
  final_features = model.fc.in_features
  layer1 = 1500
  layer2 = 700
  out = 102
  
  model_seq = nn.Sequential(
      nn.Linear(final_features,layer1),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(layer1,layer2),
      nn.ReLU(),
      nn.Linear(layer2,out),
      nn.LogSoftmax(dim=1)
  )
  
  model.fc = model_seq
  
  return model


model_conv = pretrainedmodel()
model_conv = model_conv.to(device)


optimizer = optim.Adam(filter(lambda p: p.requires_grad,model_conv.parameters()))
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


numepochs = 25

def validation_accuracy(net,loader):
  net.eval()
  with torch.no_grad():
    val_acc = 0
    total = 0
    val_loss = 0.0
    for imgs,labels in loader:
      imgs = imgs.to(device)
      labels = labels.to(device)
      
      out = net(imgs)
      
      curloss = F.nll_loss(out,labels)
      val_loss+=curloss.item() * imgs.size(0)
      
      _,preds = torch.max(out,1)
      val_acc+=torch.sum(preds==labels.data).item()
      total+=imgs.size(0)
      
    return ((val_acc/total)*100),val_loss/total


val_best_loss = np.Inf


trainsize = dataset_sizes['train']
best_model_wts = copy.deepcopy(model_conv.state_dict())
for epoch in range(numepochs):
  print("Epoch {}".format(epoch))
  train_acc = 0
  train_loss = 0.0
  model_conv.train()
  exp_lr_scheduler.step()
  for inputs,categories in dataloaders['train']:
    inputs = inputs.to(device)
    categories = categories.to(device)
    
    optimizer.zero_grad()
    
    output = model_conv(inputs)
    _,predictions = torch.max(output,1)
    loss = F.nll_loss(output,categories)
    
    loss.backward()
    optimizer.step()
    
    train_acc+=torch.sum(predictions==categories.data).item()
    train_loss+=loss.item() * inputs.size(0)
    
  train_loss = train_loss/trainsize
  train_acc = (train_acc/trainsize)*100
  
  print("Train Loss {} Train Accuracy {}".format(train_loss,train_acc))
  
  valid_accuracy,valid_loss = validation_accuracy(model_conv,dataloaders['valid'])
  
  print("Validation Loss {} Validation Accuracy {}".format(valid_loss,valid_accuracy))
  
  if(valid_loss<val_best_loss):
    val_best_loss = valid_loss
    best_model_wts = copy.deepcopy(model_conv.state_dict())

model_conv.load_state_dict(best_model_wts)


torch.save(model_conv.state_dict(),'res152_1.pt')


data_transformstest = {
    'test': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'testoriginal': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


image_datasetstest = {x: datasets.ImageFolder(os.path.join('/', x),data_transformstest[x])
                  for x in ['test']}
dataloaderstest = {x: torch.utils.data.DataLoader(image_datasetstest[x], batch_size=32,
                                             shuffle=False, num_workers=4)
              for x in ['test']}


model_conv_ = pretrainedmodel()
model_conv_.load_state_dict(torch.load('res152_1.pt'))
model_conv_ = model_conv_.to(device)

test_accuracy,test_loss = validation_accuracy(model_conv_,dataloaderstest['test'])

print(test_accuracy)


image_datasetorg = {x: datasets.ImageFolder(os.path.join('/', x),data_transformstest[x])
                  for x in ['testoriginal']}
dataloadersorg = {x: torch.utils.data.DataLoader(image_datasetorg[x], batch_size=32,
                                             shuffle=False, num_workers=4)
              for x in ['testoriginal']}


test_accuracyorg,test_lossorg = validation_accuracy(model_conv_,dataloadersorg['testoriginal'])


print(test_accuracyorg)

