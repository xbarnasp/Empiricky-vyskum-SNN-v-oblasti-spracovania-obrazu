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
from torchvision import transforms
from torchvision.datasets import Flowers102
from torchvision.models import vgg16
from torchvision.models import vgg16_bn

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

def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)

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
    run_dir = f"./FinalTests/runs_VGG16_Flowers102/run{file_id}"
    os.makedirs(run_dir, exist_ok=True)

    for batch in range(4):
        code_execution_start = time.time()
        number_of_total_epochs = 0

        seed = seed + batch
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        os.environ['PYTHONHASHSEED'] = str(seed)
        set_seed(seed)

        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        train_data = Flowers102(root="./data", split="train", transform=data_transform, download=True)
        validation_data = Flowers102(root="./data", split="val", transform=data_transform, download=True)
        test_data = Flowers102(root="./data", split="test", transform=data_transform, download=True)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        set_seed(seed)

        model = vgg16_bn(weights='IMAGENET1K_V1')

        classic_classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 102)
        )

        selu_classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.SELU(inplace=True),
            nn.AlphaDropout(p=0.2),
            nn.Linear(4096, 4096),
            nn.SELU(inplace=True),
            nn.AlphaDropout(p=0.2),
            nn.Linear(4096, 102)
        )

        if set_model == 1:
            model.classifier = classic_classifier
        elif set_model == 2:
            model.classifier = selu_classifier
        else:
            raise ValueError("set_model must be 1 (classic) or 2 (SELU + AlphaDropout)")

        model = model.to(device)

        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              momentum=momentum, weight_decay=weight_decay)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)
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
                    f'Number of trainable parameters: {num_params}\n\n'
                    f'Dataset: Flowers102',
                    file=f
                )
            f.write(f"{run_id}  -  Test accuracy for batch {batch}: {test_acc:.2f}%\n"
                    f"Number of epochs ran: {number_of_total_epochs}\n"
                    f"Duration: {(code_execution_end - code_execution_start) * 1000:.2f} ms\n")

        plot_training_curves(train_accuracies, val_accuracies, train_losses, val_losses, run_dir, batch)

if __name__ == "__main__":
    main(set_model=1)  #Základný_model
    # main(set_model=2)
    main(set_model=2, lr=0.001, batch=32) #SELU_s_AD
