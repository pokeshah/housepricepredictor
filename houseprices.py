"""
House Price Predictor for Kaggle's House Prices: Advanced Regression Techniques competition
Uses a Deep AI approach instead of traditional Random Forrest Approach
Current Accuracy: ~ $17k with only 100 epochs, 2 layers (can increase but need more compute)
"""


import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm # not sure how much of a performance hit this is since its python while everything else is c
import numpy as np

DATA_PATH_TRAIN = 'data/train.csv'
DATA_PATH_TEST = 'data/test.csv'
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 3000
DEVICE = torch.device("cpu") # Try CUDA on NVIDIA GPU or MPS on Apple Silicon (might yield faster performance)

def preprocessdata(train_path, test_path):
    """quick function for importing and preprocessing data"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y = train_df['SalePrice']
    X = train_df.drop(columns=['SalePrice'])

    
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = X[col].fillna('Missing')
        test_df[col] = test_df[col].fillna('Missing')
        X[col] = le.fit_transform(X[col].astype(str))
        test_df[col] = test_df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

    
    X.fillna(X.median(), inplace=True)
    test_df.fillna(test_df.median(), inplace=True)

    scaler = StandardScaler()
    X[X.columns] = scaler.fit_transform(X[X.columns])
    test_df[test_df.columns] = scaler.transform(test_df[test_df.columns])

    return X, y, test_df

X, y, test_df = preprocessdata(DATA_PATH_TRAIN, DATA_PATH_TEST)

target_scaler = StandardScaler()
y = target_scaler.fit_transform(y.values.reshape(-1, 1))
y = pd.Series(y.flatten())  

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# createa  house dataset class to be used in dataloader
class HouseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = HouseDataset(X_train, y_train)
val_dataset = HouseDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class HousePriceModel(nn.Module):
    def __init__(self, input_size):
        super(HousePriceModel, self).__init__()
        # TODO: Experiement with more layers (but try to prevent overfitting)
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

input_size = X_train.shape[1]
model = HousePriceModel(input_size).to(DEVICE)

criterion = nn.MSELoss() # TODO: Implement custm loss fucntion since Kaggle uses RMSD
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode='min',
#     factor=0.5,
#     patience=50,
#     min_lr=1e-6
# )

def train_model(model, train_loader, criterion, optimizer, epochs, device):
    progress_bar = tqdm(range(epochs), desc='Training Progress', leave=True)
    model.train()
    for _ in progress_bar:
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # TODO: Not a great system training and then validating every cycle, so use Scheduler to determine LR
        model.eval()
        all_preds_tensor = []
        all_targets_tensor = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                all_preds_tensor.append(outputs)
                all_targets_tensor.append(y_batch)

        all_preds_tensor = torch.cat(all_preds_tensor).cpu().numpy()
        all_targets_tensor = torch.cat(all_targets_tensor).cpu().numpy()

        all_preds = target_scaler.inverse_transform(all_preds_tensor).flatten()
        all_targets = target_scaler.inverse_transform(all_targets_tensor).flatten()

        mae = np.mean(np.abs(all_preds - all_targets))
        progress_bar.set_postfix(mae=mae, lr = scheduler.get_last_lr())

        model.train() 
        # scheduler.step(np.mean(epoch_loss/len(train_loader)))

train_model(model, train_loader, criterion, optimizer, EPOCHS, DEVICE)

model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        outputs = model(X_batch)
        all_preds.extend(outputs.cpu().numpy())
        all_targets.extend(y_batch.cpu().numpy())

all_preds = target_scaler.inverse_transform(all_preds).flatten()
all_targets = target_scaler.inverse_transform(all_targets).flatten()

print(f"RMSD: {np.sqrt(np.mean((all_preds - all_targets)**2))}") # this is the metric used in Kaggle Leaderboard