import pytest
import torch
from mnist_exercise.data.make_dataset import get_dataloaders

import os.path


@pytest.mark.skipif(not os.path.exists("data/processed/train_images.pt"), reason="Data files not found")
def test_data():
    train_loader, test_loader = get_dataloaders()
    assert len(train_loader.dataset) == 50000, "Training set did not have the correct number of samples"
    assert len(test_loader.dataset) == 5000, "Test set did not have the correct number of samples"
    unique_labels = torch.Tensor([])
    for im, lab in train_loader:
        assert im.shape[1:] == (1, 28, 28)
        unique_labels = torch.unique(torch.concat([unique_labels, lab]))
    assert torch.all(unique_labels == torch.arange(10))
    unique_labels = torch.Tensor([])
    for im, lab in test_loader:
        assert im.shape[1:] == (1, 28, 28)
        unique_labels = torch.unique(torch.concat([unique_labels, lab]))
    assert torch.all(unique_labels == torch.arange(10))
