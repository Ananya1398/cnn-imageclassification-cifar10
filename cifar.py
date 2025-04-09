import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import random

#Downloaded training data from CIFAR10
training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

#Downloaded testing data from CIFAR10
test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 16

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

#Created our first feed-forward model with ReLu activation function
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32*32*3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

#Created our second feed-forward model with Tanh activation function
class FeedforwardNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(32*32*3, 256),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

#Created our third model which is a CNN net
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.fc_stack(x)
        return x

model1 = NeuralNetwork().to(device)
print("Model 1:")
print(model1)

model2 = FeedforwardNet2().to(device)
print("Model 2:")
print(model2)

model3 = ConvNet().to(device)
print("Model 3:")
print(model3)

#Defining our loss function and optimizers
loss_fn = nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-4)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
optimizer3 = torch.optim.Adam(model3.parameters(), lr=1e-4)


#Our training function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            current_loss = loss.item()
            current = (batch + 1) * len(X)
            print(f"loss: {current_loss:>7f}  [{current:>5d}/{size:>5d}]")


#Our testing function that returns accuracy
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    accuracy = correct / size * 100
    print(f"Test Accuracy: {accuracy:.1f}% for model: {model}")
    return accuracy

#Here we evaluate training loss at the end of an epoch in eval mode
def evaluate_loss(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    size = len(dataloader.dataset)
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
    return total_loss / size


epochs = 30

#Training and testing of Model 1 - feed forward with ReLu
print(f"Training and Testing Model 1 starts here\n")
train_losses_model1 = []
for t in range(epochs):
    print(f"\nEpoch {t+1} for Model 1\n")
    train(train_dataloader, model1, loss_fn, optimizer1)
    loss = evaluate_loss(train_dataloader, model1, loss_fn)
    train_losses_model1.append(loss)
    accuracy1 = test(test_dataloader, model1, loss_fn)
    if t > 0 and train_losses_model1[-1] > train_losses_model1[-2]:
        print("Stopping early as loss increased.")
        break
print(f"Model1 trained!\n")

#Training and testing of Model 2 - feed forward with Tanh
print(f"Training Model 2 starts here\n")
train_losses_model2 = []
for t in range(epochs):
    print(f"\nEpoch {t + 1} for Model 2\n")
    train(train_dataloader, model2, loss_fn, optimizer2)
    loss = evaluate_loss(train_dataloader, model2, loss_fn)
    train_losses_model2.append(loss)
    accuracy2 = test(test_dataloader, model2, loss_fn)
    if t > 0 and train_losses_model2[-1] > train_losses_model2[-2]:
        print("Stopping early as loss increased.")
        break
print(f"Model2 trained!\n")

#Training and testing of Model 3 - CNN
print(f"Training Model 3 starts here\n")
train_losses_model3 = []
for t in range(epochs):
    print(f"\nEpoch {t+1} for Model 3\n")
    train(train_dataloader, model3, loss_fn, optimizer3)
    loss = evaluate_loss(train_dataloader, model3, loss_fn)
    train_losses_model3.append(loss)
    accuracy3 = test(test_dataloader, model3, loss_fn)
    if t > 0 and train_losses_model3[-1] > train_losses_model3[-2]:
        print("Stopping early as loss increased.")
        break
print(f"Model3 trained!\n")
print("Training Done!")

#Saving our model
torch.save(model1.state_dict(), "model1.pth")
torch.save(model2.state_dict(), "model2.pth")
torch.save(model3.state_dict(), "model3.pth")

#Plotting graph of the training loss after each epoch of training for Model 1
plt.figure()
plt.plot(train_losses_model1, label='Model 1')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss per Epoch')
plt.legend()
plt.savefig('training_loss_model1.png')
plt.clf()

#Plotting graph of the training loss after each epoch of training for Model 2
plt.figure()
plt.plot(train_losses_model2, label='Model 2')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss per Epoch')
plt.legend()
plt.savefig('training_loss_model2.png')
plt.clf()

#Plotting graph of the training loss after each epoch of training for Model 3
plt.figure()
plt.plot(train_losses_model3, label='Model 3')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss per Epoch')
plt.legend()
plt.savefig('training_loss_model3.png')
plt.clf()

'''
Here we start the code needed to identify 1 correctly predicted and 1 incorrectly predicted image
of each net.
'''
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def save_sample_predictions(model, test_data, model_name):
    model.eval()
    correct = None
    incorrect = None

    indices = list(range(len(test_data)))
    random.shuffle(indices) # We were getting the same correct image and same incorrect image for all 3 models. So added this randomness to allow our start index to be different for all 3.

    for i in indices:
        x, y = test_data[i]
        x_device = x.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x_device)
            predicted_label = torch.argmax(pred, 1).item()

        if predicted_label == y and correct is None:
            correct = (x, y, predicted_label)
        elif predicted_label != y and incorrect is None:
            incorrect = (x, y, predicted_label)

        if correct and incorrect:
            break

    def save_image(img_tensor, true_label, pred_label, label_type):
        img = img_tensor.permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.title(f"{label_type.capitalize()} Prediction\nTrue: {classes[true_label]}, Pred: {classes[pred_label]}")
        plt.axis('off')
        plt.savefig(f"{model_name}_{label_type}.png")
        plt.clf()
        print(f"{label_type.capitalize()} prediction saved as {model_name}_{label_type}.png")

    if correct:
        save_image(*correct, "correct")
    if incorrect:
        save_image(*incorrect, "incorrect")

save_sample_predictions(model1, test_data, "model1")
save_sample_predictions(model2, test_data, "model2")
save_sample_predictions(model3, test_data, "model3")

print("\nFinal Accuracies:")
print(f"Model 1 (Feedforward-ReLU): {accuracy1:.2f}%")
print(f"Model 2 (Feedforward-Tanh): {accuracy2:.2f}%")
print(f"Model 3 (CNN): {accuracy3:.2f}%")