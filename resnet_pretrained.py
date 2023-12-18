import numpy as np
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
import warnings
import os
import glob
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

import image_dataset

# Ignore warnings during execution
warnings.filterwarnings("ignore")

######################################
########## HELPER FUNCTIONS ##########
######################################

# Create topk accuracy function


def topk_accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.

    Parameters:
    - output (torch.Tensor): Model predictions (logits).
    - target (torch.Tensor): Ground truth labels.
    - topk (tuple): Tuple of integers specifying the top-k values for accuracy computation.

    Returns:
    List of accuracy values for each specified top-k value.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get top k predictions.
        _, pred = output.topk(maxk, dim=1)

        # Find the predicted classes and transpose
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(torch.round(correct_k.mul_(
                1.0 / batch_size), decimals=4))
        return res


def save_checkpoint(model, epoch, optimizer, loss, train_acc1, train_acc5, filename):
    """
    Save a checkpoint of the model, optimizer, and training metrics.

    Parameters:
    - model (torch.nn.Module): The PyTorch model.
    - epoch (int): The current epoch.
    - optimizer (torch.optim.Optimizer): The optimizer used for training.
    - loss: The current training loss.
    - train_acc1: Top-1 training accuracy.
    - train_acc5: Top-5 training accuracy.
    - filename (str): The filename for saving the checkpoint.

    Returns:
    None
    """
    checkpoint = {
        'model': model.state_dict(),
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'loss': loss,
        'train_acc1': train_acc1,
        'train_acc5': train_acc5
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, filename):
    """
    Load a previously saved model checkpoint.

    Parameters:
    - model (torch.nn.Module): The PyTorch model.
    - optimizer (torch.optim.Optimizer): The optimizer used for training.
    - filename (str): The filename of the saved checkpoint.

    Returns:
    Tuple containing the loaded model, optimizer, start epoch, current loss, top-1 training accuracy, and top-5 training accuracy.
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start = checkpoint['epoch']
    curr_loss = checkpoint['loss']
    train_acc1 = checkpoint['train_acc1']
    train_acc5 = checkpoint['train_acc5']
    return model, optimizer, start, curr_loss, train_acc1, train_acc5


def get_recent_checkpoint(checkpoint_folder='checkpoints'):
    """
    Get the filename of the most recent model checkpoint in the 'resnet_checkpoints' directory.

    Returns:
    str: The filename of the most recent checkpoint.
    """
    list_of_files = glob.glob(
        f'{checkpoint_folder}/*.pth')  # * means all, if need specific format then *.pth
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def train(epochs, checkpoint_folder, pretrained, verbose):

    start_with_checkpoint = pretrained
    if start_with_checkpoint:
        model = models.resnet18(pretrained=True)

        # Modify the first convolutional layer for grayscale images
        num_input_channels = 1  # Grayscale images have only one channel
        model.conv1 = nn.Conv2d(num_input_channels, 64,
                                kernel_size=7, stride=2, padding=3, bias=False)

        # Change the number out potential output classes.
        num_classes = 1139
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        try:
            checkpoint_name = get_recent_checkpoint(checkpoint_folder)
        except ValueError:
            print(
                'Error in start_with_checkpoint; no checkpoints found. Set start_with_checkpoint = False')

        model, optimizer, start_epoch, train_losses, train_acc1, train_acc5 = load_checkpoint(
            model, optimizer, checkpoint_name)
        print('Successfully loaded checkpoint') if verbose else None
    elif not start_with_checkpoint:
        model = models.resnet18(pretrained=True)

        # Modify the first convolutional layer for grayscale images
        num_input_channels = 1  # Grayscale images have only one channel
        model.conv1 = nn.Conv2d(num_input_channels, 64,
                                kernel_size=7, stride=2, padding=3, bias=False)

        # Change the number out potential output classes.
        num_classes = 1139
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        start_epoch = 0
        train_acc1 = []
        train_acc5 = []
        train_losses = []
        print('Starting from scratch') if verbose else None

    # Make sure the checkpoint folder exists (used for checkpointing in training loop)
    Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)

    train_dataset = image_dataset.ImageDataset(
        train=True, apply_equalize=False, apply_transform_train=True, transform=transforms.ToTensor())
    test_dataset = image_dataset.ImageDataset(
        train=False, apply_equalize=False, apply_transform_test=True, transform=transforms.ToTensor())
    print('Successfully loaded datasets') if verbose else None

    # Define the indices for the train and test sets
    dataset_size = len(train_dataset)
    split = int(0.8 * dataset_size)  # 80% for training, 20% for testing

    indices = list(range(dataset_size))
    train_indices, test_indices = indices[:split], indices[split:]

    # Create data samplers for train and test sets
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create DataLoader instances using the samplers
    batch_size = 30
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=test_sampler)
    print('Successfully loaded dataloaders') if verbose else None

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Set the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using Device', device) if verbose else None
    model.to(device)

    # Training loop
    num_epochs = epochs + start_epoch
    print('Starting training loop') if verbose else None

    for epoch in range(start_epoch, num_epochs):
        model.train()
        tqdm_dataloader = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        epoch_loss = 0.0  # Initialize the epoch loss

        for batch in tqdm_dataloader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(labels)  # Accumulate loss for the entire epoch

            tqdm_dataloader.set_postfix(loss=loss.item())

        # Evaluate the model's accuracy
        model.eval()
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        topk = (1, 5)  # You can adjust the top-k values as needed

        for batch in tqdm(test_dataloader):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1)

            # Compute top-k accuracy
            topk_acc = topk_accuracy(outputs, labels, topk=topk)

            correct_top1 += topk_acc[0].item()
            correct_top5 += topk_acc[1].item()

            tqdm_dataloader.set_postfix(OrderedDict({'loss': loss.item(
            ), 'top1_acc': topk_acc[0].item(), 'top5_acc': topk_acc[1].item()}))

            total += len(labels)

        # Calculate overall accuracy
        accuracy_top1 = correct_top1 / total
        accuracy_top5 = correct_top5 / total
        train_acc1.append(accuracy_top1)
        train_acc5.append(accuracy_top5)
        # tqdm_dataloader.set_postfix(loss=epoch_loss / len(train_dataloader.dataset), top1_acc=accuracy_top1, top5_acc=accuracy_top5)

        # Close the tqdm progress bar for the epoch
        tqdm_dataloader.close()

        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(train_dataset)
        # Save the average epoch loss for plotting
        train_losses.append(avg_epoch_loss)

        # Save the model
        save_checkpoint(model, epoch, optimizer, train_losses, train_acc1,
                        train_acc5, f"{checkpoint_folder}/resnet18_epoch{epoch}.pth")

    # Training complete
    print("Training complete!")

    fig, ax = plt.subplots(1, 2, figsize=(12, 8), dpi=200)
    ax[0].plot(train_losses, label='Training Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training Loss over Epochs')
    ax[0].legend()

    ax[1].plot(train_acc1, label='Top 1 Accuracy')
    ax[1].plot(train_acc5, label='Top 5 Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Training Accuracy over Epochs')
    ax[1].legend()

    plt.savefig('resnet_acc_loss.png')


if __name__ == '__main__':
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(
        description='Arguments for ResNet Training.')

    # Define command-line arguments
    parser.add_argument('--training_epochs', default=5,
                        type=int, help='Number of epochs to train for.')
    parser.add_argument('--checkpoint_folder', type=str,
                        help='Folder to store checkpoint files.')
    parser.add_argument('--pretrained', default=True,
                        type=bool, help='Use checkpointed ResNet model.')
    parser.add_argument('--verbose', default=False,
                        help='Print verbose output.')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the arguments
    epochs = args.training_epochs
    checkpoint_folder = args.checkpoint_folder
    pretrained = args.pretrained
    print('Pretrained:', pretrained)
    verbose = args.verbose

    train(epochs, checkpoint_folder, pretrained, verbose)
    # print('Done!')
