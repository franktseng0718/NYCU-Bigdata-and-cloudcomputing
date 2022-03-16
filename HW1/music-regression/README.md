# Music Memorability Prediction

## Usage
```
pip install -r requirements.txt
python music_regression_cnn.py
```

## Result

score=0.009

## Code structure
### Extract feature and save in npy
```code=python
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
mels = librosa.feature.melspectrogram(y=y, sr=sr)
mels = librosa.power_to_db(mels, ref=np.max)

```
```code=python
if train:
        y_train = np.array(y_train)
        mfcc_data = np.array(mfcc_train)
        melspec_data = np.array(melspec_train)
        np.save('y_train', y_train)
        np.save('mfcc_train', mfcc_data)
        np.save('melspec_train', melspec_data)
else:
        mfcc_data = np.array(mfcc_test)
        melspec_data = np.array(melspec_test)
        np.save('mfcc_test', mfcc_data)
        np.save('melspec_test', melspec_data)
```

### Import libraries
```code=python
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import cv2
# Preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, TensorDataset
import torchvision.models as models
import torchvision.transforms as transforms
```
### Load data and normalize
```code=python
X = np.load('melspec_train.npy')
X = np.expand_dims(X, axis=1)
X = np.resize(X, (220, 1, 128, 216))
#X = np.resize(X, (220, 3, 128, 216))
print(X.shape)
y = np.load('y_train.npy')
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[1]*X.shape[2]*X.shape[3])).reshape(X.shape)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
X_train, y_train = X, y
print(len(y_train))
print(len(y_val))
X_train.shape
X_train = torch.from_numpy(np.array(X_train)).float()
y_train = torch.from_numpy(np.array(y_train)).float()
X_val = torch.from_numpy(np.array(X_val)).float()
y_val = torch.from_numpy(np.array(y_val)).float()
X = CustomTensorDataset(tensors=(X_train, y_train))
Valid = CustomTensorDataset(tensors=(X_val, y_val))
```
```code=python
X_test = np.load('melspec_test.npy')
X_test = np.expand_dims(X_test, axis=1)
X_test = np.resize(X_test, (15, 1, 128, 216))
#X_test = np.resize(X_test, (15, 3, 128, 216))
scaler = StandardScaler()

X_test = scaler.fit_transform(X_test.reshape(-1, X_test.shape[1]*X_test.shape[2]*X_test.shape[3])).reshape(X_test.shape)
X_test.shape
X_test = torch.from_numpy(np.array(X_test)).float()
y_test = torch.FloatTensor(X_test.shape[0])
Test = CustomTensorDataset(tensors=(X_test, y_test))

```
```code=python
trainLoader = torch.utils.data.DataLoader(X, batch_size=4, shuffle=True, num_workers=2)
valLoader = torch.utils.data.DataLoader(Valid, batch_size=1, shuffle=False, num_workers=2)
testLoader = torch.utils.data.DataLoader(Test, batch_size=1, shuffle=False, num_workers=2)
```
### Model architecture
```code=python
# LENet
# Model structure
cfg = [8, 16, 32, 32, 32]
#cfg = [4, 4]
class LENet(nn.Module):
    def __init__(self):
        super(LENet, self).__init__()
        self.features = self._make_layers(cfg)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)


        return x
    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2)]
            in_channels = x
        return nn.Sequential(*layers)
```
### Init optimizer
```code=python
Conv_model = LENet()
"""
Conv_model = models.resnet50(pretrained=True)
for param in Conv_model.parameters():
    param.requires_grad = False

new_fc = nn.Sequential(*list(Conv_model.fc.children())[:-1] + [nn.Linear(2048, 1)])
Conv_model.fc = new_fc
"""
Conv_model = Conv_model.cuda()
# loss
criterion = nn.MSELoss(reduction='sum')
# optimizer
learning_rate = 1e-3
weight_decay = 0
optimizer = torch.optim.Adam(Conv_model.parameters(), lr = learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
```

### Train and Val
```code=python
def train(model):
    validation_error = []
    patience = 100
    last_loss = 1e9
    trigger_times = 0
    total_epoch = 1000
    for epoch in range(total_epoch):
        running_loss = 0.0
        for times, data in enumerate(trainLoader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            #print(inputs.shape)
            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            outputs = outputs.reshape(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if times+1 == len(trainLoader):
                print('[%d/%d, %d/%d] loss: %.3f' % (epoch+1, total_epoch, times+1, len(trainLoader), running_loss))
        validation_error.append(val(valLoader, model))
        print('validation_error:', validation_error[-1])
        if validation_error[-1] > last_loss:
            trigger_times += 1
            print('trigger times:', trigger_times)
        else:
            trigger_times = 0
            print('trigger times: 0')
        if trigger_times >= patience:
            print('Early stopping!\nStart to test process.')
            break
        last_loss = validation_error[-1]
    print('Finished Training')
```
```code=python
def val(valLoader, model):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        for data in valLoader:
            inputs, label = data
            inputs, label = inputs.to(device), label.to(device)
            MLP_model = model.cuda()
            outputs = model(inputs)
            outputs = outputs[0][0]
            #print('label:', label)
            total_loss += np.abs(np.sum((label - outputs).cpu().numpy()))
        return total_loss
        
```
### Save model's weight
```code=python
PATH = './Conv_Net_melspec.pth'
torch.save(Conv_model.state_dict(), PATH)
```

### Inference
```code=python
# inference
predicted_res = []
with torch.no_grad():
    net.eval()
    for data in testLoader:
        input, label = data
        input = input.to(device)
        output = net(input)
        output = output.cpu().numpy()
        predicted_res.append(output[0][0])
submission_df = pd.read_csv('sample_submission.csv')
submission_df['score'] = predicted_res

#submission_df = submission_df.drop(['score'], axis=1)
#submission_df = submission_df.insert(1, "score", predicted_res)
submission_df.to_csv('submission/CNN_submission_melspec.csv', index=False)

```




