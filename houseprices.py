import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

# Configuration
DATA_PATH_TRAIN = 'data/train.csv'
DATA_PATH_TEST = 'data/test.csv'
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 1000

# Device configuration
DEVICE = torch.device("cpu") # or "mps" if available

# Data Loading and Preprocessing
def load_and_preprocess_data(train_path, test_path):
    """Loads and preprocesses the training and test data."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y = train_df['SalePrice']
    X = train_df.drop(columns=['SalePrice'])

    # Categorical Feature Encoding
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = X[col].fillna('Missing')
        test_df[col] = test_df[col].fillna('Missing')
        X[col] = le.fit_transform(X[col].astype(str))
        test_df[col] = test_df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

    # Numerical Feature Imputation and Scaling
    X.fillna(X.median(), inplace=True)
    test_df.fillna(test_df.median(), inplace=True)

    scaler = StandardScaler()
    X[X.columns] = scaler.fit_transform(X[X.columns])
    test_df[test_df.columns] = scaler.transform(test_df[test_df.columns])

    return X, y, test_df

X, y, test_df = load_and_preprocess_data(DATA_PATH_TRAIN, DATA_PATH_TEST)

# Data Splitting
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Dataset and DataLoader
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

# Model Definition
class HousePriceModel(nn.Module):
    def __init__(self, input_size):
        super(HousePriceModel, self).__init__()
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

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Function
def train_model(model, train_loader, criterion, optimizer, epochs, device):
    """Trains the model."""
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
        progress_bar.set_postfix(loss=epoch_loss / len(train_loader))

# Model Training
train_model(model, train_loader, criterion, optimizer, EPOCHS, DEVICE)