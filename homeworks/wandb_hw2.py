import wandb
import torch
from PIL import Image
from sklearn.metrics import recall_score, precision_score
from torchvision import datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np


class BaseTransform:
    def __init__(self, p: float):
        self.p = p

    def __call__(self, image: Image.Image):
        raise NotImplementedError()


class RandomCrop(BaseTransform):
    def __init__(self, crop_size: int):
        super().__init__(1)
        self.crop_size = crop_size

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if height == self.crop_size and width == self.crop_size:
            return image
        top = torch.randint(0, height - self.crop_size, [1]).item()
        left = torch.randint(0, width - self.crop_size, [1]).item()
        bottom = top + self.crop_size
        right = left + self.crop_size
        return image.crop((left, top, right, bottom))


class RandomRotate(BaseTransform):
    def __init__(self, p: float, max_angle: int):
        super().__init__(p)
        self.max_angle = max_angle

    def __call__(self, image: Image.Image) -> Image.Image:
        if torch.rand(1).item() < self.p:
            angle = torch.rand(1).item() * self.max_angle
            return image.rotate(angle)
        return image


class RandomZoom(BaseTransform):
    def __init__(self, p: float, zoom_factor: float):
        super().__init__(p)
        self.zoom_factor = zoom_factor

    def __call__(self, image: Image.Image) -> Image.Image:
        if torch.rand(1).item() < self.p:
            width, height = image.size
            new_width = int(width * self.zoom_factor)
            new_height = int(height * self.zoom_factor)
            return image.resize((new_width, new_height))
        return image


class ToTensor(BaseTransform):
    def __init__(self):
        super().__init__(1)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        width, height = image.size
        channels = len(image.getbands())
        image = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        image = image.view(height, width, channels)
        image = image.permute(2, 0, 1)
        return image / 255


class Compose:
    def __init__(self, transforms: list[BaseTransform]):
        self.transforms = transforms

    def __call__(self, image: Image.Image):
        for transform in self.transforms:
            image = transform(image)
        return image


class my_model(nn.Module):
    def __init__(self):
        super(my_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        return x


def train(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        preds = model(inputs)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = preds.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / size

    print(f'Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    wandb.log({"train accuracy": accuracy, "train loss": avg_loss})


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, num_correct = 0, 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            preds = model(inputs)
            num_correct += (preds.argmax(1) == targets).type(torch.float).sum().item()
            test_loss += loss_fn(preds, targets).item()

            # Store predictions and targets for metrics computation
            all_preds.extend(preds.argmax(1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_loss /= num_batches

    # Compute precision and recall
    precision = 100 * precision_score(all_targets, all_preds, average='weighted', zero_division=np.nan)
    recall = 100 * recall_score(all_targets, all_preds, average='weighted', zero_division=np.nan)
    accuracy = 100 * num_correct / size

    print(f'Test Error, Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n'
          f'Precision: {precision:.2f}%, Recall: {recall:.2f}%')

    wandb.log(
        {"test accuracy": accuracy, "test loss": test_loss, "test recall": recall, "test precision": precision})

    return test_loss


lrs = [0.0001, 0.001, 0.01]
rotate_ps = [0, 0.25, 0.5, 0.75, 1]
zoom_ps = [0, 0.25, 0.5, 0.75, 1]
epochs = 10
device = "cuda" if torch.cuda.is_available() else "cpu"


def run(lr, epochs, rand_rotate_p, rand_rotate_max_angle, rand_zoom_p, rand_zoom_factor):
    wandb.init(
        # set the wandb project where this run will be logged
        project="Testing Transforms",

        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "architecture": "CNN",
            "dataset": "FashionMNIST",
            "epochs": epochs,
            "random rotate p": rand_rotate_p,
            "random rotate max angle": rand_rotate_max_angle,
            "random zoom p": rand_zoom_p,
            "random zoom factor": rand_zoom_factor
        })

    transforms = Compose([
        RandomZoom(p=rand_zoom_p, zoom_factor=rand_zoom_factor),
        RandomRotate(p=rand_rotate_p, max_angle=rand_rotate_max_angle),
        RandomCrop(crop_size=28),
        ToTensor()
    ])

    train_data = datasets.FashionMNIST(
        root="../notebooks/data",
        train=True,
        transform=transforms,
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="../notebooks/data",
        train=False,
        download=True,
        transform=transforms
    )

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True,
                              collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False,
                             collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

    model = my_model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for t in range(epochs):
        train(train_loader, model, loss_fn, optimizer, t)
        test(test_loader, model, loss_fn)
        print('-------------------------------------------')

    wandb.finish()

i = 0
for lr in lrs:
    for rotate_p in rotate_ps:
        for zoom_p in zoom_ps:
            print(f'Ready {i / 75 * 100:2f}%')
            run(lr, epochs, rotate_p, 30, zoom_p, 1.25)
