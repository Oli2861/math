import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

print(torch.__version__)


def load_cifar(batch_size: int = 4, path: str = "./data"):
    assert os.path.exists(path), f"Path {path} does not exist."

    # Transform to tensors of normalized range [-1, 1]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = CIFAR10(root=path, train=True, download=True, transform=transform)
    testset = CIFAR10(root=path, train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    return trainloader, testloader, classes


class ConvolutionalNeuralNetwork(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)  # kernel_size, stride
        self.conv1 = nn.Conv2d(3, 6, 5)  # in_channel, out_channel, kernel_size
        self.conv2 = nn.Conv2d(6, 16, 5)  # in_channel, out_channel, kernel_size
        self.lin1 = nn.Linear(16 * 5 * 5, 120)  # in_features, out_features
        self.lin2 = nn.Linear(120, 84)  # in_features, out_features
        self.lin3 = nn.Linear(84, 10)  # in_features, out_features

    def forward(self, x):
        # First convolutional layer: Convolution -> ReLu -> Pool
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Second convolutional layer: Convolution -> ReLu -> Pool
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)  # Flatten all except batch dimension

        # Linear layers: Linear layer -> ReLu
        x = self.lin1(x)
        x = F.relu(x)

        x = self.lin2(x)
        x = F.relu(x)

        return self.lin3(x)


def train_cnn(
    trainloader: DataLoader,
    cnn: nn.Module,
    criterion,
    optimizer,
    epochs: int = 2,
):
    for epoch in range(epochs):
        running_loss = 0.0

        for index, data in enumerate(trainloader, 0):
            x, y = data

            optimizer.zero_grad()  # Reset gradients to 0

            y_pred = cnn(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if index % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {index + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0


def test_cnn(testloader: DataLoader, cnn: nn.Module):
    correct = 0
    count = 0

    with torch.no_grad():
        for data in testloader:
            x, y = data

            # Obtain probability distribution over classes
            y_pred = cnn(x)
            # Choose class with highest probability as prediction
            _, y_pred = torch.max(y_pred.data, 1)

            count += y.size(0)
            correct += (y_pred == y).sum().item()

    accuracy = 100 * correct // count
    assert accuracy > 10, "CNN does not better than random guessing (assuming 10 classes)"
    print(f"Accuracy of the network on the 10000 test images: {accuracy} %")


if __name__ == "__main__":
    trainloader, testloader, classes = load_cifar()

    cnn = ConvolutionalNeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

    train_cnn(trainloader, cnn, criterion, optimizer)
    test_cnn(testloader, cnn)
