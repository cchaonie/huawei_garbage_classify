#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import sys
import copy
from PIL import Image
import random
from glob import glob
import codecs
import math

save_dir_obs = '../../output/model'

data_dir = "../../train_data"

model_local = "../../pretrained"

num_classes = 40

batch_size = 128

num_workers = 4

num_epochs = 10

learning_rate = 5e-3

momentum = .9

input_size = 224

test_percent = .3

def init_model(model_name, state_path, num_classes):
    model = models.resnet18()
    if model_name == 'resnet50':
        model = models.resnet50()
    model.load_state_dict(torch.load(state_path))
    for param in model.parameters():
        param.requires_grad = False
    num_fc_if = model.fc.in_features
    model.fc = nn.Linear(num_fc_if, num_classes)
    return model

# model = init_model('resnet18', os.path.join(model_local, 'resnet18-5c106cde.pth'), num_classes)
model = init_model('resnet50', os.path.join(model_local, 'resnet50-19c8e357.pth'), num_classes)

class GarbageDataSet(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        super(GarbageDataSet, self).__init__()
        self.imgs = data
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

def check_data(data_dir):
    label_files = glob(os.path.join(data_dir, '*.txt'))
    random.shuffle(label_files)
    img_paths = []
    labels = []
    for _, file_path in enumerate(label_files):
        with codecs.open(file_path, 'r', 'utf-8') as f:
            line = f.readline()
        line_split = line.strip().split(', ')
        if len(line_split) != 2:
            print('%s contain error label' % os.path.basename(file_path))
            continue
        img_name = line_split[0]
        label = int(line_split[1])
        img_paths.append(os.path.join(data_dir, img_name))
        labels.append(label)
    return img_paths, labels


img_paths, labels = check_data(data_dir)
print(len(img_paths), len(labels))

label_dict = dict()
for i, l in enumerate(labels):
    str_l = str(l)
    l_count = label_dict.get(str_l)
    if l_count is None:
        l_count = []
    l_count.append(i)
    label_dict[str_l] = l_count

def custom_split(img_paths, labels):
    test_labels = []
    train_labels = []
    test_paths = []
    train_paths = []
    for val in label_dict.values():
        len_val = len(val)
        rand_index = random.choices(
            range(len_val), k=math.floor(len_val * test_percent))
        for i, label_i in enumerate(val):
            if i in rand_index:
                test_paths.append(img_paths[label_i])
                test_labels.append(labels[label_i])
            else:
                train_paths.append(img_paths[label_i])
                train_labels.append(labels[label_i])
    print('train data count: %d, test data count: %d' % (len(train_paths), len(test_paths)))
    results1 = []
    results2 = []
    for i in range(len(train_paths)):
        results1.append((train_paths[i], train_labels[i]))
    for i in range(len(test_paths)):
        results2.append((test_paths[i], test_labels[i]))
    return results1, results2


train_data, val_data = custom_split(img_paths, labels)
# print(train_data[0], val_data[0])

composed_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_transforms = {
    'train': composed_transform,
    'val': composed_transform,
}

print("Initializing Datasets and Dataloaders...")
image_datasets = {
    'train': GarbageDataSet(data=train_data, transform=data_transforms['train']),
    'val': GarbageDataSet(data=val_data, transform=data_transforms['val'])
}

dataloaders_dict = {
    x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    ) for x in ['train', 'val']
}
print("Initialization finished...")

print("GPU count: %d" % torch.cuda.device_count())

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

params_to_update = model.parameters()
print("Params to learn:")
params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t", name)

optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum)
criterion = nn.CrossEntropyLoss()

def train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for index, (inputs, labels) in enumerate(dataloaders[phase]):
                if phase == 'train':
                    print('batch: %d, left: %d, percent: %6.4f' % (index,
                        float(len(dataloaders[phase].dataset)) / batch_size - 1 - index, 
                        float((index + 1) * batch_size) / len(dataloaders[phase].dataset)))
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    labels = labels.to(outputs.device)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss=running_loss / len(dataloaders[phase].dataset)
            epoch_acc=running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc=epoch_acc
                best_model_wts=copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
        print()
    time_elapsed=time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# Train and evaluate
model, history=train_model(
    model,
    dataloaders_dict,
    criterion,
    optimizer
)

def save_model(model, save_dir):
    torch.save(model.state_dict(), os.path.join(
        save_dir, 'trained_model_state.pth'))
    print('save model_state success')

save_model(model, save_dir_obs)
