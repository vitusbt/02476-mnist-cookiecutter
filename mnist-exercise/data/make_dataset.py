import sys
import os
import torch
from torch.utils.data import TensorDataset, DataLoader

if __name__ == '__main__':
    # Get the data and process it

    train_images_0 = torch.load('data/raw/train_images_0.pt')
    train_images_1 = torch.load('data/raw/train_images_1.pt')
    train_images_2 = torch.load('data/raw/train_images_2.pt')
    train_images_3 = torch.load('data/raw/train_images_3.pt')
    train_images_4 = torch.load('data/raw/train_images_4.pt')
    train_images_5 = torch.load('data/raw/train_images_5.pt')
    train_target_0 = torch.load('data/raw/train_target_0.pt')
    train_target_1 = torch.load('data/raw/train_target_1.pt')
    train_target_2 = torch.load('data/raw/train_target_2.pt')
    train_target_3 = torch.load('data/raw/train_target_3.pt')
    train_target_4 = torch.load('data/raw/train_target_4.pt')
    train_target_5 = torch.load('data/raw/train_target_5.pt')

    test_images = torch.load('data/raw/test_images.pt').unsqueeze(1)
    test_target = torch.load('data/raw/test_target.pt')

    train_images = torch.concat([
        train_images_0,
        train_images_1,
        train_images_2,
        train_images_3,
        train_images_4,
        train_images_5
    ]).unsqueeze(1)

    #train_images = train_images*2 - 1
    #test_images = test_images*2 - 1

    train_mean = torch.mean(train_images)
    train_std = torch.std(train_images)

    train_images = (train_images - train_mean) / train_std
    test_images = (test_images - train_mean) / train_std

    train_target = torch.concat([
        train_target_0,
        train_target_1,
        train_target_2,
        train_target_3,
        train_target_4,
        train_target_5
    ])

    torch.save(train_images, "data/processed/train_images.pt")
    torch.save(test_images, "data/processed/test_images.pt")
    torch.save(train_target, "data/processed/train_target.pt")
    torch.save(test_target, "data/processed/test_target.pt")


def get_dataloaders(batch_size=64):
    train_images = torch.load("data/processed/train_images.pt")
    test_images = torch.load("data/processed/test_images.pt")
    train_target = torch.load("data/processed/train_target.pt")
    test_target = torch.load("data/processed/test_target.pt")

    train_dataset = TensorDataset(train_images, train_target)
    test_dataset = TensorDataset(test_images, test_target)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader
