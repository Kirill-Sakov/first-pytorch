import time

import torch
from torch import nn
from torch.utils.data import DataLoader, default_collate
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def collate_fn(x):
    return tuple(x_.to(device) for x_ in default_collate(x))


batch_size = 64
train_data_loader = DataLoader(training_data, batch_size=batch_size, collate_fn=collate_fn)
test_data_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


nn_model = NeuralNetwork().to(device)
epochs = 10
learning_rate = 1e-3
loss_function = nn.CrossEntropyLoss()
nn_optimizer = torch.optim.SGD(nn_model.parameters(), lr=learning_rate)


def train_loop(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    for batch, (X, y) in enumerate(data_loader):
        # Prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(data_loader, model, loss_fn):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in data_loader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")


start_time = time.process_time()
for t in range(epochs):
    print(f"Epoch {t+1}\n--------------------------------")
    train_loop(train_data_loader, nn_model, loss_function, nn_optimizer)
    test_loop(test_data_loader, nn_model, loss_function)
stop_time = time.process_time()
print(f"Done! Time: {stop_time - start_time}")

