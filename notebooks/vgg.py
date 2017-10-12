
# coding: utf-8

# In[19]:

import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import sys
import torch.utils.data as data
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


# In[20]:


torch.cuda.is_available()


# In[21]:


# Hyperparameters
num_epochs = 700
batch_size = 64
learning_rate = 1e-5
valid_ratio = .25


# In[22]:


tensor_files = os.listdir('/home/cblythe2/github/neuralMusic/data/spectrogram_tensors/')
# len(tensor_files)


# In[23]:


df = pd.read_csv('/home/cblythe2/github/neuralMusic/data/tensor_genres.csv', dtype=object)
#df.head()


# In[24]:


class FmaDataset(data.Dataset):
    """Dataset wrapping Free Music Archive Spectrogram Tensors.
    Arguments:
        A CSV file path with the tensors, labels
        Path to tensor folder
        Extension of images
    """
    def __init__(self, csv_path, tensor_path, transform=None):
        tmp_df = pd.read_csv(csv_path, dtype=object)
        tmp_df2 = tmp_df[tmp_df['tensor_name'].apply(lambda x: np.fromfile(tensor_path + x).size==262144)]
        assert tmp_df2['tensor_name'].apply(lambda x: os.path.isfile(tensor_path + x)).all(), "Some tensors referenced in the CSV file were not found"
        assert tmp_df2['tensor_name'].apply(lambda x: np.fromfile(tensor_path + x).size==262144).all(), "Some tensors referenced had the wrong number of elements"
        tmp_df2 = tmp_df2.reset_index(drop=True)
        tmp_df2 = tmp_df2[:7552] # for clean batch divisibility

        self.mlb = MultiLabelBinarizer()
        self.tensor_path = tensor_path
        self.transform = transform

        self.X_train = tmp_df2['tensor_name']
        self.y_train = self.mlb.fit_transform(tmp_df2['genre_top'].str.split()).astype(np.float32)

    def __getitem__(self, index):
        tensor = np.fromfile(self.tensor_path + self.X_train[index])
        tensor = torch.from_numpy(tensor)
        tensor = tensor.view(512, 512)
        tensor = tensor.unsqueeze(0).float()
        if self.transform is not None:
            tensor = self.transform(tensor)

        label = self.y_train[index].argmax(axis=0)
        return tensor, label

    def __len__(self):
        return len(self.X_train.index)


# In[25]:


dataset = FmaDataset(csv_path='/home/cblythe2/github/neuralMusic/data/tensor_genres.csv', tensor_path='/home/cblythe2/github/neuralMusic/data/spectrogram_tensors/')


# In[26]:


# print(dataset.X_train.shape, dataset.y_train.shape)
# print()
# dataset.y_train[75:83]


# In[27]:


indices = torch.randperm(len(dataset))
valid_size = int(len(dataset) * valid_ratio)
train_indices = indices[:len(indices)-valid_size]
valid_indices = indices[len(indices)-valid_size:]


# In[28]:


train_loader = data.DataLoader(dataset,
                          batch_size=batch_size, sampler=data.sampler.SubsetRandomSampler(train_indices),
                          num_workers=1
                          # pin_memory =True # CUDA only
                         )

valid_loader = data.DataLoader(dataset, sampler=data.sampler.SubsetRandomSampler(valid_indices),
                          batch_size=batch_size,
                          num_workers=1
                          # pin_memory =True # CUDA only
                         )


# ## Convolutional Neural Network ~ 36% Accuracy

# In[29]:


# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
#         self.layer5 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
        self.fc = nn.Linear(128*32*32, 8)
#         self.fc = nn.Linear(256*16*16, 8)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
#         out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# In[30]:


model = CNN()
model.cuda();


# In[31]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[32]:


def train(epoch):
    model.train()
    for i, (tensors, labels) in enumerate(train_loader):
        tensors = Variable(tensors).cuda()
        labels = Variable(labels).cuda()
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(tensors)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return(loss.data[0])


# In[33]:


def test(best_accuracy):
    my_labels = []
    my_predictions = []
    model.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for tensors, labels in valid_loader:
        tensors = Variable(tensors, volatile=True).cuda()
        outputs = model(tensors)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
        my_labels += labels.tolist()
        my_predictions += predicted.cpu().tolist()
    epoch_accuracy = (100 * correct / total)
    return(epoch_accuracy)


# In[36]:


best_epoch = 0
best_accuracy = 0
print('------Hyperparameters------\nNumber of Epochs | {:.0f}\nBatch Size       | {:.0f}\nLR:              | {:.5f}\nValidation Ratio | {:.3f}\n'.format(num_epochs, batch_size, learning_rate, valid_ratio))
print("---------------------------\nTraining on {} examples\nTesting on {} examples\n---------------------------\n".format(len(train_indices), len(valid_indices)))
for epoch in range(num_epochs):
    try:
        # Train model over epoch
        loss_data = train(epoch)
        print('Epoch [%d/%d] Training Loss: %.4f' % (epoch, num_epochs, loss_data))
        # Test the Model at end of every epoch
        valid_accuracy = test(epoch)
        print("Epoch [%d/%d] Validation Accuracy: %.2f" % (epoch, num_epochs, valid_accuracy))
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pkl')
    except KeyboardInterrupt:
        print("\nBest Validation Accuracy of {:.2f}% at Epoch {:.0f}".format(best_accuracy, best_epoch))
        sys.exit(1)
print("\nBest Validation Accuracy of {:.2f}% at Epoch {:.0f}".format(best_accuracy, best_epoch))

