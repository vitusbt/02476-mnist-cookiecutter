import click
import torch

from data.make_dataset import get_dataloaders

@click.command()
@click.option('--model', help="path to model file")
@click.option('--bs', default=64, help="batch size for loading (not really important)")
def predict(
    model: torch.nn.Module,
    bs: int
) -> torch.Tensor:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        bs: batch size
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """

    print("Evaluating like my life dependends on it")

    model = torch.load(model)
    model.eval()
    _, test_set = get_dataloaders(bs)

    outputs = None

    with torch.no_grad():
        correct = 0
        total = 0
        for step, (images, labels) in enumerate(test_set):
            y = model(images)
            if outputs is None:
                outputs = y.clone()
            else:
                outputs = torch.concat([outputs, y], dim=0)
            labels_pred = torch.argmax(y, dim=1)
            correct += torch.sum(labels_pred == labels).item()
            total += labels.shape[0]

        acc = correct/total
        print(f'Correct: {correct}')
        print(f'Total: {total}')
        print(f'Accuracy: {acc*100}%')

    return outputs

    #return torch.cat([model(batch) for batch in dataloader], 0)

if __name__ == '__main__':
    predict()