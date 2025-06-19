import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

model = SmallCNN()

transform = T.ToTensor()
train_dataset = MNIST(root="./", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

def train(model, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

train(model)
model.eval()

# -------- 3. Evolution Parameters -------- #
target_class = 9
population_size = 50
mutation_rate = 0.1
generations = 150
k = 10

def random_image():
    return torch.randint(0, 256, (1, 28, 28), dtype=torch.uint8).float() / 255.0

def mutate(image, rate):
    new = image.clone()
    mask = (torch.rand_like(new) < rate).float()
    noise = torch.randint(0, 256, new.shape).float() / 255.0
    return (1 - mask) * new + mask * noise

@torch.no_grad()
def evaluate(image):
    x = image.unsqueeze(0)
    out = model(x)
    prob = F.softmax(out, dim=1)[0, target_class].item()
    return prob


population = [random_image() for _ in range(population_size)]

for gen in range(generations):
    scores = [evaluate(img) for img in population]
    topk = sorted(zip(population, scores), key=lambda x: -x[1])[:k]
    best_images = [x[0] for x in topk]
    best_score = topk[0][1]
    print(f"Gen {gen:03d} | Best confidence: {best_score:.4f}")

    if gen % 50 == 0:
        plt.imshow(best_images[0][0].detach().numpy(), cmap='gray')
        plt.title(f"Gen {gen} - Conf: {best_score:.2f}")
        plt.show()

    # Create new population
    population = []
    for img in best_images:
        population.append(img)
        for _ in range(population_size // len(best_images) - 1):
            population.append(mutate(img, mutation_rate))
