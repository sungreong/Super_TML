import torch.nn as nn
from torchvision import models
import numpy as np
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
device = "cpu"

def load_model(model_name=None,target_n=2):
    # Model selection
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, target_n)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(1024, target_n)
    model = nn.DataParallel(model.to(device))
    return model

# ----- Data to Image Transformer -----

def data2img(arr, font_size=50, resolution=(256, 256), font=cv2.FONT_HERSHEY_SIMPLEX,dataset="iris",n_columns = 2):
    """ Structured Tabular Data to Image with cv2
        NOTE currently supports only iris and wine dataset
    """
    x, y = resolution
    n_colums, n_features = n_columns, len(arr)
    n_lines = n_features % n_colums + int(n_features / n_colums)
    frame = np.ones((*resolution, 3), np.uint8)*0
    k = 0
    # ----- iris -----
    if dataset=='iris':
        for i in range(n_colums):
            for j in range(n_lines):
                try:
                    cv2.putText(
                        frame, str(arr[k]), (5+i*(x//n_colums), 5+(j+1)*(y//(n_lines+1))),
                        fontFace=font, fontScale=0.5, color=(255, 255, 255), thickness=2)
                    k += 1
                except IndexError:
                    break

    # ----- wine -----
    elif dataset=='wine':
        for i in range(n_colums):
            for j in range(n_lines):
                try:
                    cv2.putText(
                        frame, str(arr[k]), (30+i*(x//n_colums), 5+(j+1)*(y//(n_lines+1))),
                        fontFace=font, fontScale=0.4, color=(255, 255, 255), thickness=1)
                    k += 1
                except IndexError:
                    break
    elif dataset=='adult':
        for i in range(n_colums):
            for j in range(n_lines):
                try:
                    cv2.putText(
                        frame, str(arr[k]), (30+i*(x//n_colums), 5+(j+1)*(y//(n_lines+1))),
                        fontFace=font, fontScale=1.0, color=(255, 255, 255), thickness=2)
                    k += 1
                except IndexError:
                    break
    elif dataset=='boston':
        for i in range(n_colums):
            for j in range(n_lines):
                try:
                    cv2.putText(
                        frame, str(arr[k]), (30+i*(x//n_colums), 5+(j+1)*(y//(n_lines+1))),
                        fontFace=font, fontScale=0.9, color=(255, 255, 255), thickness=2)
                    k += 1
                except IndexError:
                    break
    elif dataset=='regression':
        for i in range(n_colums):
            for j in range(n_lines):
                try:
                    cv2.putText(
                        frame, str(arr[k]), (30+i*(x//n_colums), 5+(j+1)*(y//(n_lines+1))),
                        fontFace=font, fontScale=0.4, color=(255, 255, 255), thickness=1)
                    k += 1
                except IndexError:
                    break
                    
    return np.array(frame, np.uint8)


# ----- Dataset -----

class CustomTensorDataset(Dataset):
    def __init__(self, data, transform=None,dataset=None,n_columns=None):
        self.data = data
        self.transform = transform
        self.dataset = dataset
        self.n_columns= n_columns 

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        x = self.data[0][index]
        img = data2img(x,
                       dataset=self.dataset,
                       n_columns = self.n_columns)
        if self.transform:
            x = self.transform(img)

        y = self.data[1][index]
        return x, y
    
# ----- Load Data Pipeline -----

def load_data(dataset=None, batch_size=None, 
              val_size=None, test_size=None, device='cpu'):
    # load dataset
    if dataset=='iris':
        data = datasets.load_iris()
    elif dataset=='wine':
        data = datasets.load_wine()

    # Split dataset -- Cross Vaidation
    x_train, x_test, y_train, y_test \
        = train_test_split(data.data, data.target, test_size=test_size, random_state=1)

    x_train, x_val, y_train, y_val \
        = train_test_split(x_train, y_train, test_size=val_size, random_state=1)


    # Dataset and Dataloader settings
    kwargs = {} if device=='cpu' else {'num_workers': 2, 'pin_memory': True}
    loader_kwargs = {'batch_size':batch_size, **kwargs}

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Build Dataset
    train_data = CustomTensorDataset(data=(x_train, y_train), transform=transform)
    val_data   = CustomTensorDataset(data=(x_val, y_val), transform=transform)
    test_data  = CustomTensorDataset(data=(x_test, y_test), transform=transform)

    # Build Dataloader
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_data, shuffle=True, **loader_kwargs)
    test_loader  = DataLoader(test_data, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader

@torch.no_grad()
def valid_step(model, criterion, val_loader, type_ = "regression"):
    model.eval()
    avg_loss , metric = 0 , 0
    for i, (x_imgs, labels) in enumerate(val_loader):
        # forward pass
        x_imgs, labels = x_imgs.to(device), labels.to(device)
        outputs = model(x_imgs)
        if type_ == "classification" :
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            metric += torch.sum(preds == labels.data).item()
        elif type_ == "regression" :
            metric += criterion(outputs.double(), labels.double()).item()
    return {'loss': metric / len(val_loader), 'metric': metric / len(val_loader.dataset)}


def train_step(model, criterion, optimizer, train_loader, type_ = "regression"):
    model.train()
    avg_loss, metric = 0 , 0
    for i, (x_imgs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        # forward pass
        x_imgs, labels = x_imgs.to(device), labels.to(device)
        probs = model(x_imgs)
        if type_ == "classification" :
            loss = criterion(probs, labels)
        elif type_ == "regression" :
            loss = criterion(probs.double(), labels.double())
        # back-prop
        loss.backward()
        optimizer.step()
        # gather statistics
        avg_loss += loss.item()
        if type_ == "classification" :
            loss = criterion(probs, labels)
            _, preds = torch.max(probs, 1)
            metric += torch.sum(preds == labels.data).item()
        elif type_ == "regression" :
            metric += criterion(probs.double(), labels.double()).item()
    return {'loss': avg_loss / len(train_loader), 'metric': metric / len(train_loader.dataset)}


def opt_selection(model, opt=None):
    if opt=='Adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=0.0001)
    elif opt=='Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001, weight_decay=1e-5)
    elif opt=='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    else:
        raise NotImplementedError
    return optimizer



def train_model(model_name='densenet121', opt='Adagrad', dataset='iris', writer=None):
    train_loader, val_loader, test_loader = load_data(dataset)

    # Model selection
    model = load_model(model_name)

    # Optimizer
    optimizer = opt_selection(model, opt)

    # Loss Criterion
    criterion = nn.CrossEntropyLoss()

    best_train, best_val = 0.0, 0.0
    for epoch in range(1, epochs+1):
        # Train and Validate
        train_stats = train_step(model, criterion, optimizer, train_loader)
        valid_stats = valid_step(model, criterion, val_loader)

        # Logging
        logging(epoch, train_stats, valid_stats, writer)

        # Keep best model
        if valid_stats['accuracy'] > best_val or (valid_stats['accuracy']==best_val and train_stats['accuracy']>=best_train):
            best_train  = train_stats['accuracy']
            best_val    = valid_stats['accuracy']
            best_model_weights = copy.deepcopy(model.state_dict())

    # Load best model and evaluate on test set
    model.load_state_dict(best_model_weights)
    test_stats = valid_step(model, criterion, test_loader)

    print('\nBests Model Accuracies: Train: {:4.2f} | Val: {:4.2f} | Test: {:4.2f}'.format(best_train, best_val, test_stats['accuracy']))

    return model