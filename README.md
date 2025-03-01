# House Price Predictor
House Price Predictor for Kaggle's House Prices: Advanced Regression Techniques competition

## Implementation
Uses PyTorch and Deep NN to solve the problem, iterating on the Random Forest Approach. Model Architecture:
```py
nn.Linear(input_size, 128),
nn.ReLU(),
nn.Dropout(0.2),
nn.Linear(128, 64),
nn.ReLU(),
nn.Dropout(0.5),
nn.Linear(64, 1)
```

## Usage
`python3 houseprices.py` after installing dependencies; Config Variables are in the code
