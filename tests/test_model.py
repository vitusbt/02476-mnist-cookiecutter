import torch
import pytest
from mnist_exercise.models.model import MyAwesomeModel


def test_model():
    model = MyAwesomeModel()
    for bs in [1, 17, 32, 100]:
        input = torch.randn(size=(bs, 1, 28, 28))
        output = model(input)
        assert output.shape == (bs, 10)


# tests/test_model.py
def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match="Expected input to a 4D tensor"):
        model(torch.randn(1, 2, 3))
    with pytest.raises(ValueError, match="Expected each sample to have shape *"):
        model(torch.randn(1, 2, 3, 4))
