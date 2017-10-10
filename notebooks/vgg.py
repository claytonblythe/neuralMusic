
# coding: utf-8

# In[1]:

import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch.utils.data as data
from sklearn.preprocessing import MultiLabelBinarizer


# In[2]:

torch.cuda.is_available()


# In[3]:

num_epochs = 300
batch_size = 64
learning_rate = 0.001


# In[4]:

tensor_files = os.listdir('/home/cblythe2/github/neuralMusic/data/spectrogram_tensors/')


# In[5]:

len(tensor_files)


# In[6]:

df = pd.read_csv('/home/cblythe2/github/neuralMusic/data/tensor_genres.csv', dtype=object)
df.shape


# In[7]:

# def filter_tensor_files(tensor_path, tensor_file):
#     tensor = np.fromfile(tensor_path + tensor_file)
#     tensor = torch.from_numpy(tensor)
#     tensor = tensor.view(-1, 512)
#     if tensor.shape[0] != 512 or tensor.shape[1] != 512:
#         return(False)
#     else:
#         return(True)


# In[8]:

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
        tmp_df2 = tmp_df2[:7552]
        
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


# In[9]:

fma_dataset = FmaDataset(csv_path='/home/cblythe2/github/neuralMusic/data/tensor_genres.csv', tensor_path='/home/cblythe2/github/neuralMusic/data/spectrogram_tensors/')


# In[10]:

indices = torch.randperm(len(fma_dataset))
valid_size = int(len(fma_dataset) * .25)
train_indices = indices[:len(indices)-valid_size]
valid_indices = indices[len(indices)-valid_size:]


# In[11]:

fma_dataset.X_train.shape


# In[12]:

fma_dataset.y_train.shape


# In[13]:

fma_dataset.y_train[75:83]


# In[14]:

fma_dataset.X_train[0:10]


# In[ ]:




# In[15]:

train_loader = data.DataLoader(fma_dataset,
                          batch_size=batch_size, sampler=data.sampler.SubsetRandomSampler(train_indices),
                          num_workers=1
                          # pin_memory =True # CUDA only
                         )

valid_loader = data.DataLoader(fma_dataset, sampler=data.sampler.SubsetRandomSampler(valid_indices),
                          batch_size=batch_size,
                          num_workers=1
                          # pin_memory =True # CUDA only
                         )


# ## Two-layer Convolutional Neural Network

# In[16]:

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
        self.fc = nn.Linear(128*32*32, 8)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# In[17]:

cnn = CNN()
cnn.cuda()


# In[18]:

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)


# In[19]:

best_epoch = 0
best_accuracy = 0
# Train the Model
for epoch in range(num_epochs):
    for i, (tensors, labels) in enumerate(train_loader):
        tensors = Variable(tensors).cuda()
        labels = Variable(labels).cuda()
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(tensors)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #print(loss.data[0])
        if i % 50 == 0:
             print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' % (epoch+1, num_epochs, i+1, len(fma_dataset)//batch_size, loss.data[0]))
    
    # Test the Model at end of every epoch
    my_labels = []
    my_predictions = []
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for tensors, labels in valid_loader:
        tensors = Variable(tensors).cuda()
        outputs = cnn(tensors)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
        my_labels += labels.tolist()
        my_predictions += predicted.cpu().tolist()
    epoch_accuracy = (100 * correct / total)
    if epoch_accuracy > best_accuracy:
        best_epoch = epoch
        best_accuracy = epoch_accuracy
        torch.save(cnn.state_dict(), 'best_cnn.pkl')
    print("Epoch {} Validation Accuracy: {}").format(epoch, epoch_accuracy)
    
print("Best Validation Accuracy of {}% at Epoch {}").format(best_accuracy, best_epoch)


# In[20]:

# my_labels = []
# my_predictions = []

# # Test the Model
# cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
# correct = 0
# total = 0
# for tensors, labels in train_loader:
#     tensors = Variable(tensors).cuda()
#     outputs = cnn(tensors)
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted.cpu() == labels).sum()
#     my_labels += labels.tolist()
#     my_predictions += predicted.cpu().tolist()


# In[21]:

# print('Test Accuracy of the model on the %d test songs: %d%%\n' % (len(train_indices), 100 * correct / total))
# print('%d out of %d classified correctly' % (correct, total))
# # Save the Trained Model
# torch.save(cnn.state_dict(), 'cnn.pkl')


# In[22]:

# my_labels = []
# my_predictions = []

# # Test the Model
# #cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
# correct = 0
# total = 0
# for tensors, labels in valid_loader:
#     tensors = Variable(tensors).cuda()
#     outputs = cnn(tensors)
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted.cpu() == labels).sum()
#     my_labels += labels.tolist()
#     my_predictions += predicted.cpu().tolist()


# In[23]:

# print('Test Accuracy of the model on the %d test songs: %d%%\n' % (len(valid_indices), 100 * correct / total))
# print('%d out of %d classified correctly' % (correct, total))
# # Save the Trained Model
# torch.save(cnn.state_dict(), 'cnn.pkl')

