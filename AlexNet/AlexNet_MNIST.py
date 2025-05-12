import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import datetime
import time
import matplotlib.pyplot as plt
import random
import math
import pandas as pd

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
from torchvision import datasets, transforms
from pathlib import Path

DATA_DIR = Path("MNIST")

if not torch.cuda.is_available():
    raise Exception("CUDA is not available. Please ensure you have a GPU with CUDA support.")

device = torch.device('cuda')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lecun_normal_init(tensor):
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(1.0 / fan_in)

    with torch.no_grad():
        return tensor.normal_(0.0, std)

class AlexNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.features3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2304, 410),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(410, 410),
            nn.ReLU(inplace=True),
            nn.Linear(410, 10)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                lecun_normal_init(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                lecun_normal_init(m.weight)
                nn.init.zeros_(m.bias)


class AlexNet_BachNorm(nn.Module):

    def __init__(self):
        super().__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.features3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.features4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.features5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2304, 410),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(410, 410),
            nn.ReLU(inplace=True),
            nn.Linear(410, 10)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                lecun_normal_init(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                lecun_normal_init(m.weight)
                nn.init.zeros_(m.bias)


class AlexNet_SNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.features3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
        )
        self.features4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
        )
        self.features5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2304, 410),
            nn.SELU(inplace=True),
            nn.Linear(410, 410),
            nn.SELU(inplace=True),
            nn.Linear(410, 10)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                lecun_normal_init(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                lecun_normal_init(m.weight)
                nn.init.zeros_(m.bias)

class AlexNet_SNN_AD(nn.Module):

    def __init__(self):
        super().__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.features3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
        )
        self.features4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
        )
        self.features5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.AlphaDropout(0.2),
            nn.Linear(2304, 410),
            nn.SELU(inplace=True),
            nn.AlphaDropout(0.2),
            nn.Linear(410, 410),
            nn.SELU(inplace=True),
            nn.Linear(410, 10)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                lecun_normal_init(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                lecun_normal_init(m.weight)
                nn.init.zeros_(m.bias)


def plot_training_curves(train_accuracies, val_accuracies, train_losses, val_losses, run_dir, batch):
    plt.figure()
    plt.plot(train_accuracies, label='Tréningová presnosť')
    plt.plot(val_accuracies, label='Validačná presnosť')
    plt.title('Tréningová a validačná presnosť')
    plt.xlabel('Epochy')
    plt.ylabel('Presnosť (%)')
    plt.legend()
    plt.savefig(os.path.join(run_dir, f"accuracy_plot_batch{batch}.png"))
    plt.close()

    plt.figure()
    plt.plot(train_losses, label='Tréningová strata')
    plt.plot(val_losses, label='Validačná strata')
    plt.title('Tréningová a validačná strata')
    plt.xlabel('Epochy')
    plt.ylabel('Strata')
    plt.legend()
    plt.savefig(os.path.join(run_dir, f"loss_plot_batch{batch}.png"))
    plt.close()

    results_df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train_Loss': train_losses,
        'Val_Loss': val_losses,
        'Train_Accuracy': train_accuracies,
        'Val_Accuracy': val_accuracies
    })
    results_csv_path = os.path.join(run_dir, f"training_results_batch{batch}.csv")
    results_df.to_csv(results_csv_path, index=False)


def train_one_epoch(model, train_loader, optimizer, criterion):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 30 == 0:
            print(f"  [Train] Processed {batch_idx * len(images)}/{len(train_loader.dataset)} samples")

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, val_loader, criterion):

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    epoch_loss = val_loss / len(val_loader.dataset)
    epoch_acc = 100.0 * val_correct / val_total
    return epoch_loss, epoch_acc


def test_model(model, test_loader, criterion):

    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = 100.0 * test_correct / test_total
    return test_loss, test_acc

def check_dataset_mean(data_loader):
    mean_sum = torch.zeros(1).to(device)
    total_pixels = 0

    print("Checking dataset mean after normalization...")

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            batch_pixels = images.shape[0] * images.shape[2] * images.shape[3]
            mean_sum += images.mean(dim=[0, 2, 3]) * batch_pixels
            total_pixels += batch_pixels

    dataset_mean = mean_sum / total_pixels
    print(f"Mean of the dataset after normalization: {dataset_mean.item()}")


def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)

    first_batch = next(iter(loader))
    images, _ = first_batch
    channels = images.shape[1]

    mean = torch.zeros(channels).to(device)
    std = torch.zeros(channels).to(device)
    total_pixels = 0

    print("Computing dataset mean and standard deviation...")

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            batch_pixels = images.shape[0] * images.shape[2] * images.shape[3]
            mean += images.mean(dim=[0, 2, 3]) * batch_pixels
            std += images.std(dim=[0, 2, 3]) * batch_pixels
            total_pixels += batch_pixels

    mean /= total_pixels
    std /= total_pixels

    print(f"Computed Mean: {mean}")
    print(f"Computed Std: {std}")

    return mean.cpu().tolist(), std.cpu().tolist()

def main(lr=0.01, set_model=1, batch=128):

    seed = 42
    learning_rate = lr
    momentum = 0.9
    weight_decay = 0.0005
    batch_size = batch
    number_of_epochs = 200
    patience = 3

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    set_seed(seed)

    run_id = str(datetime.datetime.today())
    file_id = run_id.replace(" ", "_").replace(".", "").replace(":", "_")
    run_dir = f"./FinalTests/runs_AlexNet_MNIST/run{file_id}"
    os.makedirs(run_dir, exist_ok=True)

    for batch in range(2):
        code_execution_start = time.time()
        number_of_total_epochs = 0

        classes = os.listdir(str(DATA_DIR / 'test'))
        validation_path = str(DATA_DIR / 'validation')
        for c in classes:
            os.makedirs(os.path.join(validation_path, c), exist_ok=True)


        data_path_val = DATA_DIR / 'validation'
        data_path_validation = list(data_path_val.glob("*/*.png"))
        for path in data_path_validation:
            os.replace(path, str(path).replace('/validation', '/train'))

        if set_model > 2:
            temp_dataset = datasets.ImageFolder(root=str(DATA_DIR / 'train'), transform=transforms.ToTensor())
            computed_mean, computed_std = compute_mean_std(temp_dataset)

        train_data_paths = list(DATA_DIR.glob("train/*/*.png"))
        for i in range(len(train_data_paths)):
            if (((batch+6) * 200) % 6000) <= (i % 6000) < ((((batch+6) + 1) * 200) % 6000):
                move_dir = str(train_data_paths[i]).replace('/train', '/validation')
                os.replace(str(train_data_paths[i]), move_dir)

        if set_model > 2:
            data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=computed_mean, std=computed_std)])
        else:
            data_transform = transforms.Compose([transforms.ToTensor()])

        test_data = datasets.ImageFolder(
            root=str(DATA_DIR / 'test'),
            transform=data_transform
        )
        train_data = datasets.ImageFolder(
            root=str(DATA_DIR / 'train'),
            transform=data_transform
        )
        validation_data = datasets.ImageFolder(
            root=str(DATA_DIR / 'validation'),
            transform=data_transform
        )

        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            validation_data, batch_size=batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False
        )

        if set_model == 1:
            model = AlexNet().to(device)
        elif set_model == 2:
            model = AlexNet_BachNorm().to(device)
        elif set_model == 3:
            model = AlexNet_SNN().to(device)
        else:
            model = AlexNet_SNN_AD().to(device)

        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )

        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=1, verbose=True
        )

        criterion = nn.CrossEntropyLoss()

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters in the model: {num_params}")

        trigger_times = 0
        best_loss = np.inf

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in tqdm(range(number_of_epochs), desc=f"Batch {batch} Epochs"):
            number_of_total_epochs += 1

            epoch_loss, epoch_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc = validate_one_epoch(model, val_loader, criterion)

            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            print(
                f"\nEpoch {epoch + 1}/{number_of_epochs}, "
                f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n"
            )

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate after epoch {epoch + 1}: {current_lr}")

            if val_loss < best_loss:
                best_loss = val_loss
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print('Early stopping!')
                    break

        test_loss, test_acc = test_model(model, test_loader, criterion)
        code_execution_end = time.time()
        print(f"Test accuracy for batch {batch}: {test_acc:.2f}%")

        with open(os.path.join(run_dir, "results_model.txt"), "a") as f:
            if batch == 0:
                print(model, file=f)
                print(
                    f'\n\n\nLearning_rate = {learning_rate}\n'
                    f'Weight_decay = {weight_decay}\n'
                    f'Batch_size = {batch_size}\n'
                    f'Number_of_set_epochs = {number_of_epochs}\n'
                    f'Optimizer: {optimizer}\n'
                    f'Scheduler: ReduceLROnPlateau(mode=min, factor=0.1, patience=1, verbose=True)\n'
                    f'Number of trainable parameters: {num_params}\n\n',
                    f'Dataset: {DATA_DIR}',
                    file=f
                )
            f.write(f"{run_id}  -  Test accuracy for batch {batch}: {test_acc:.2f}%\n"
                    f"Number of epochs ran: {number_of_total_epochs}\n"
                    f"Duration: {(code_execution_end - code_execution_start) * 1000:.2f} ms\n")


        plot_training_curves(train_accuracies, val_accuracies, train_losses, val_losses, run_dir, batch)

        data_path_val = DATA_DIR / 'validation'
        data_path_validation = list(data_path_val.glob("*/*.png"))
        for path in data_path_validation:
            os.replace(path, str(path).replace('/validation', '/train'))


if __name__ == "__main__":
    main(lr=0.01, set_model=1)  #Basic_model
    main(lr=0.01, set_model=2)  #BN_model
    main(lr=0.001, set_model=3)  #SNN_model
    main(lr=0.001, set_model=4, batch=32)  #SNN_model_s_AD