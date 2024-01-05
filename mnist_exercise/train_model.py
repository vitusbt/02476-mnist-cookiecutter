import sys
import os
import click

#sys.path.append('mnist-exercise/')
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# os.environ['PYTHONPATH'] = "/"

# print(sys.path)
# print('---')
# print(os.environ['PYTHONPATH'])

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from mnist_exercise.models.model import MyAwesomeModel
from mnist_exercise.data.make_dataset import get_dataloaders

@click.command()
@click.option('--lr', default=1e-3, help="learning rate to use for training")
@click.option('--bs', default=128, help="batch size")
@click.option('--epochs', default=20, help="number of epochs")
def train(lr, bs, epochs):
    """Train a model on MNIST."""

    print("Training day and night")
    print(f"lr={lr}")

    model = MyAwesomeModel()
    train_set, _ = get_dataloaders(batch_size=bs)

    # Optimizers require the parameters to optimize and a learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()

    training_losses = np.zeros(epochs)

    for ep in range(epochs):
        running_loss = 0
        for step, (images, labels) in enumerate(train_set):
            optimizer.zero_grad()
            labels_pred = model(images)
            loss = loss_fn(labels_pred, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        else:
            training_losses[ep] = running_loss/len(train_set)
            print(f"Epoch {ep+1}/{epochs} | Training loss: {training_losses[ep]}")
    else:
        torch.save(model, 'models/model_mnist.pt')

        if not os.path.exists('reports/figures'):
            os.makedirs('reports/figures')

        plt.figure()
        plt.plot(np.arange(1,epochs+1), training_losses)
        plt.savefig("reports/figures/loss_curve.png")
        plt.close()


if __name__ == "__main__":
    train()