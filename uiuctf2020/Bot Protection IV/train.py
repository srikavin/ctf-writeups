import json
import os
import string

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torchvision.transforms import Compose, ToTensor, Resize


class CaptchaDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.dataset = [os.path.join(root_dir, i) for i in os.listdir(root_dir) if '.png' in i]
        self.mapping = {k: v for v, k in enumerate(string.ascii_uppercase)}
        self.inv_mapping = {k: v for k, v in enumerate(string.ascii_uppercase)}
        self.transform = Compose([Resize((224, 224)), ToTensor()])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_name = self.dataset[idx]
        label = os.path.basename(img_name).split('_')[0]

        image = Image.open(img_name)

        ret = []
        for char in label:
            row = [0] * 26
            row[self.mapping[char]] = 1
            ret.append(row)

        image = self.transform(image)

        return (image.cuda(), torch.tensor(ret).cuda())

    def get_label_string(self, label):
        batch = []
        for i in range(label.shape[0]):
            cur = []

            for row in range(label.shape[1]):
                max_idx = label[i][row].argmax()
                cur.append(self.inv_mapping[max_idx.item()])

            batch.append(''.join(cur))

        return batch


dataset = CaptchaDataset('captchas')

model = models.resnet18()
model.fc = nn.Linear(512, 5 * 26)
model = model.cuda()

lengths = [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)]
train_split, valid_split = random_split(dataset, lengths)

batch_size = 64
train_dataloader = DataLoader(train_split, batch_size=batch_size, drop_last=True, shuffle=True)
valid_dataloader = DataLoader(valid_split, batch_size=batch_size, drop_last=True, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.8, 0.999))


def compute_loss(prediction, label):
    loss = 0
    for i in range(5):
        loss += F.cross_entropy(prediction[:, i], label[:, i].argmax(1)).mean()
    return loss


metrics = {'train': [], 'valid': [], 'loss': []}
for epoch in range(1):
    epoch_metrics = {'train': [], 'valid': [], 'loss': []}

    model.train()
    for idx, (image, label) in enumerate(train_dataloader):
        optimizer.zero_grad()
        prediction = model(image).view(-1, 5, 26)
        loss = compute_loss(prediction, label)
        loss.backward()
        optimizer.step()

        # use numpy to speed up comparisons
        predicted_labels = np.array(dataset.get_label_string(prediction))
        actual_labels = np.array(dataset.get_label_string(label))
        accuracy = (predicted_labels == actual_labels).sum() / prediction.shape[0]
        epoch_metrics['train'].append(accuracy)
        epoch_metrics['loss'].append(loss.item())

        print(f"Epoch {epoch + 1} Batch {idx + 1} Loss {loss.item():.2f} Accuracy {accuracy:.2f}", end='\r')

    model.eval()
    for idx, (image, label) in enumerate(valid_dataloader):
        prediction = model(image).view(-1, 5, 26)
        predicted_labels = np.array(dataset.get_label_string(prediction))
        actual_labels = np.array(dataset.get_label_string(label))
        accuracy = (predicted_labels == actual_labels).sum() / prediction.shape[0]
        epoch_metrics['valid'].append(accuracy)

    metrics['train'].extend(epoch_metrics['train'])
    metrics['valid'].extend(epoch_metrics['valid'])
    metrics['loss'].extend(epoch_metrics['loss'])
    print(f"Epoch {epoch + 1} complete.")
    print(f"Training Loss: {np.mean(epoch_metrics['loss'][-1]):.2f}")
    print(f"Training Accuracy: {np.mean(epoch_metrics['train'][-1]):.2f}")
    print(f"Validation Accuracy: {np.mean(epoch_metrics['valid'][-1]):.2f}")

torch.save(model, "resnet_state")

json.dump(metrics, open('metrics.json', 'w'))
