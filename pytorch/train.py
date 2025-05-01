import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from model import MLP
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

### -------- HYPERPARAMETERS ------- ###

file_path = "../monte-carlo/data/vg_prices_all.csv"
input_size = 5
hidden_size = 100
output_size = 1
num_epochs = 1000
batch_size = 5000
learning_rate = 1e-4

### --------- CUDA SETTINGS -------- ###

assert torch.cuda.is_available(), "CUDA is not available"
device = torch.device("cuda")
print(f"Using device: {device}")

### ------------ DATASET ----------- ###

df = pd.read_csv(file_path)

X, y = df[["T", "K", "kappa", "sigma", "theta"]], df["price"]

X, y = torch.tensor(X.values, dtype=torch.float32), torch.tensor(
    y.values, dtype=torch.float32
)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")

X_train, X_val = X_train.to(device), X_val.to(device)
y_train, y_val = y_train.to(device), y_val.to(device)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train),
    batch_size=batch_size,
    shuffle=True,
)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_val, y_val), batch_size=batch_size
)

### ---------- TRAINING ---------- ###

model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(
    device
)
criterion = nn.MSELoss(reduction="sum")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Training model...")
train_losses, val_losses = model.train_model(
    train_loader, val_loader, criterion, optimizer, num_epochs
)
print("Model trained")
print(f"Training loss: {train_losses[-1]}")
print(f"Validation loss: {val_losses[-1]}")

print("Saving model...")
torch.save(model.state_dict(), "models/model.pth")
print("Model saved to models/model.pth")

### ------------ PLOTTING ------------ ###

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot 1: Training and Validation Loss
ax1.plot(train_losses, label="Training Loss")
ax1.plot(val_losses, label="Validation Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_yscale("log")
ax1.set_title("Training and Validation Loss Over Time")
ax1.legend()
ax1.grid(True)

# Get predictions for validation set
model.eval()
with torch.no_grad():
    val_predictions = model(X_val).cpu().numpy()
    actual_prices = y_val.cpu().numpy()

# Plot 2: Predicted vs Actual Prices
ax2.scatter(actual_prices, val_predictions, alpha=0.5)
ax2.plot(
    [0, actual_prices.max()],
    [0, actual_prices.max()],
    "r--",
    label="Perfect Prediction",
)
ax2.set_xlabel("Actual Price")
ax2.set_ylabel("Predicted Price")
ax2.set_title("Predicted vs Actual Prices on Val Set")
ax2.legend()
ax2.grid(True)

# Adjust layout and save
plt.tight_layout()
plt.savefig("visualizations/training_plots.png")
print("Training plots saved to visualizations/training_plots.png")
