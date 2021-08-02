import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models
import torchvision.transforms as transforms

import os
import numpy as np
from torchvision.transforms.transforms import RandomHorizontalFlip, RandomResizedCrop

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

batch_size = 20
num_workers = 2
epochs = 20

# Resize the images to 224 x 224 and normalize them
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load data into memory
data_dir = './data/'
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')
test_dir = os.path.join(data_dir, 'test')

train_data = datasets.ImageFolder(train_dir, transform=train_transform)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transform)
test_data = datasets.ImageFolder(test_dir, transform=test_transform)

# A dictionary that maps type to loader
loaded_dict = {
    'train': torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True),
    'valid': torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=True),
    'test': torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
}


# Convolutional Neural Network set up
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # A list of layers
        # First convolution starting with 3 channels in
        # 32 3 x 3 filters with padding size and stride of 1
        self.l1 = nn.Conv2d(3, 32, (3, 3), padding=1, stride=1)
        # 64 3 x 3 filters(number of filters * 2 every layer)
        self.l2 = nn.Conv2d(32, 64, (3, 3), padding=1, stride=1)
        self.l3 = nn.Conv2d(64, 128, (3, 3), padding=1, stride=1)
        self.l4 = nn.Conv2d(128, 128, (3, 3), padding=1, stride=1)

        # Max pool with size 2 x 2 at the end
        self.pooling = nn.MaxPool2d(2, 2)

        # Apply dropouts
        self.fc1 = nn.Linear(25088, 5000)
        self.fc2 = nn.Linear(5000, 512)
        self.fc3 = nn.Linear(512, 133)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Apply ReLU activation function
        x = F.relu(self.l1(x))
        # Pool
        x = self.pooling(x)
        x = F.relu(self.l2(x))
        x = self.pooling(x)
        x = F.relu(self.l3(x))
        x = self.pooling(x)
        x = F.relu(self.l4(x))
        x = self.pooling(x)
        # Flatten
        x = x.view(-1, 25088)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


modelNet = Net()

# Calculate loss and use adam optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(modelNet.parameters(), lr=0.0001)

# train model


def train(model, loaded_dict, optimizer, loss_func, epochs, saved):
    min_valid = np.Inf
    for epoch in range(1, epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for batch_idx, (data, target) in enumerate(loaded_dict['train']):
            output = model(data)
            loss = loss_func(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + \
                ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        # Validation
        model.eval()
        for batch_idx, (data, target) in enumerate(loaded_dict['valid']):
            output = model(data)
            loss = loss_func(output, target)
            valid_loss = valid_loss + \
                ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
        ))

        if valid_loss <= min_valid:
            print(
                'Validation loss decreased (({:.6f} --> {:.6f}). Saving...'.format(min_valid, valid_loss))
            torch.save(model.state_dict(), saved)
            min_valid = valid_loss
    return model


modelNet = train(modelNet, loaded_dict, optimizer,
                 loss_func, epochs, 'model_scratch.pt')


modelNet.load_state_dict(torch.load('model_scratch.pt'))

# Test the model


def test(loaded_dict, model, loss_func):
    test_loss = 0.0
    correct = 0.0
    total = 0.0

    model.eval()
    for batch_idx, (data, target) in enumerate(loaded_dict["test"]):
        output = model(data)
        loss = loss_func(output, target)
        test_loss = test_loss + ((1 / (batch_idx + 1))
                                 * (loss.data - test_loss))
        pred = output.data.max(1, keepdim=True)[1]

        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' %
          (100. * correct / total, correct, total))


test(loaded_dict, modelNet, loss_func)
