import re
import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
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
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'valid']}



dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


num_classes = 102


def pretrainedmodel():
  model = torchvision.models.resnet50(pretrained=True)
  for param in model.parameters():
    param.requires_grad = False
  
  final_features = model.fc.in_features
  
  layer1 = 1200
  layer2 = 600
  out = 102
  
  model_seq = nn.Sequential(
      nn.Linear(final_features,layer1),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(layer1,layer2),
      nn.ReLU(),
      nn.Dropout(0.4),
      nn.Linear(layer2,out)
  )
  
  model.fc = model_seq
  
  return model


model_conv = pretrainedmodel()

model_conv = model_conv.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad,model_conv.parameters()),lr=0.001,momentum=0.9)


numepochs = 30


def validation_accuracy(net,loader):
  with torch.no_grad():
    net.eval()
    val_acc = 0
    total = 0
    val_loss = 0.0
    for imgs,labels in loader:
      imgs = imgs.to(device)
      labels = labels.to(device)
      
      out = net(imgs)
      
      curloss = criterion(out,labels)
      val_loss+=curloss.item() * imgs.size(0)
      
      _,preds = torch.max(out,1)
      val_acc+=torch.sum(preds==labels.data).item()
      
    total = dataset_sizes['valid']
      
    return ((val_acc/total)*100),val_loss/total



model_conv.train()
val_best = 0
trainsize = dataset_sizes['train']
best_model_wts = copy.deepcopy(model_conv.state_dict())
for epoch in range(numepochs):
  print("Epoch {}".format(epoch))
  train_acc = 0
  train_loss = 0.0
  for inputs,categories in dataloaders['train']:
    inputs = inputs.to(device)
    categories = categories.to(device)
    
    optimizer.zero_grad()
    
    output = model_conv(inputs)
    _,predictions = torch.max(output,1)
    loss = criterion(output,categories)
    
    loss.backward()
    optimizer.step()
    
    train_acc+=torch.sum(predictions==categories.data).item()
    train_loss+=loss.item() * inputs.size(0)
    
  train_loss = train_loss/trainsize
  train_acc = (train_acc/trainsize)*100
  
  print("Train Loss {} Train Accuracy {}".format(train_loss,train_acc))
  
  valid_accuracy,valid_loss = validation_accuracy(model_conv,dataloaders['valid'])
  
  print("Validation Loss {} Validation Accuracy {}".format(valid_loss,valid_accuracy))
  
  if(valid_accuracy>val_best):
    val_best = valid_accuracy
    best_model_wts = copy.deepcopy(model_conv.state_dict())

model_conv.load_state_dict(best_model_wts)


checkpoint = {'input_size': 224,
              'output_size': num_classes,
              'hidden_layers': [each for each in model_conv.children()],
              'state_dict': model_conv.state_dict()}
torch.save(checkpoint, 'resnet50_1.pt')


def load_checkpoint(checkpoint_path):
  checkpoint = torch.load(checkpoint_path,map_location=lambda storage, loc: storage)
  model = pretrainedmodel()
  return model


model_conv = load_checkpoint('resnet50_1.pt')
