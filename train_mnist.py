import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST

# noise
from torchvision.transforms import Compose, GaussianBlur, RandomRotation, ToTensor
from tqdm import tqdm

from vit import VIT, VITClassifier

writer = SummaryWriter()


def get_model():
    model = VITClassifier(
        shape=(1, 28, 28),
        n_patches_w=7,
        n_patches_h=7,
        hidden_dim=64,
        out_dim=10,
        n_blocks=4,
        n_heads=2,
        encoder_mlp_ratio=5,
    )
    return model


def train(model, train_data, val_data, n_epochs):
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(n_epochs):
        model.train()
        for i, (x, y) in tqdm(
            enumerate(train_data), total=len(train_data), desc=f"Epoch {epoch}"
        ):
            optimizer.zero_grad()
            y_hat = model(x.cuda() if cuda_available else x)
            loss = criterion(y_hat, y.cuda() if cuda_available else y)
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/train", loss, epoch * len(train_data) + i)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_data:
                if cuda_available:
                    x, y = x.cuda(), y.cuda()
                y_hat = model(x)
                _, predicted = torch.max(y_hat.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        accuracy = correct / total
        writer.add_scalar("Accuracy/val", accuracy, epoch)
        print(f"Epoch {epoch}, accuracy {accuracy}")
    writer.flush()


def main(args):
    transform = Compose([ToTensor(), GaussianBlur(3), RandomRotation(30)])
    train_data = torch.utils.data.DataLoader(
        MNIST(root="./data", train=True, download=True, transform=transform),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_data = torch.utils.data.DataLoader(
        MNIST(root="./data", train=False, download=True, transform=transform),
        batch_size=args.batch_size,
        shuffle=False,
    )
    model = get_model()
    train(model, train_data, val_data, args.n_epochs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
