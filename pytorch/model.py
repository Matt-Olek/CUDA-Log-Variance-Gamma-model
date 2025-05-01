import torch
from tqdm import tqdm
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def train_model(self, train_loader, val_loader, criterion, optimizer, num_epochs):
        train_losses = []
        val_losses = []

        for _ in tqdm(range(num_epochs)):

            self.train()
            epoch_train_loss = 0
            for batch_X, batch_y in train_loader:
                outputs = self(batch_X)
                train_loss = criterion(outputs, batch_y.view(-1, 1))

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                epoch_train_loss += train_loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            self.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self(batch_X)
                    val_loss = criterion(outputs, batch_y.view(-1, 1))
                    epoch_val_loss += val_loss.item()

            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

        return train_losses, val_losses
