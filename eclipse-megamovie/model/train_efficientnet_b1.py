# File: root/model/train_efficientnet_b1.py

import os
import time
import copy
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, Subset
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Training function
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Training mode
            else:
                model.eval()   # Evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward and optimize
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Step the scheduler
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.item())

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if validation accuracy improves
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save the model
                torch.save(model.state_dict(), 'best_model.pth')

        print()

    print(f'Best Validation Accuracy: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, train_accuracies, val_accuracies

# Custom Dataset for test data without labels
class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_paths = sorted([
            os.path.join(test_dir, fname)
            for fname in os.listdir(test_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_path  # Return image and path

if __name__ == '__main__':
    # Determine the device to run on (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Define data transformations for training, validation, and testing
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(240),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(240),
            transforms.ToTensor(),
            normalize,
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(240),
            transforms.ToTensor(),
            normalize,
        ]),
    }

    # Paths relative to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, '..'))  # Up one directory to 'eclipse-megamovie'
    train_dir = os.path.join(project_dir, 'grouped_images')
    test_dir = os.path.join(project_dir, 'test')

    # Load the full dataset
    full_dataset = datasets.ImageFolder(root=train_dir)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f'Classes: {class_names}')

    # Split the dataset into training and validation sets using stratified split
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    indices = list(range(len(full_dataset)))
    targets = [label for _, label in full_dataset.samples]

    for train_idx, val_idx in stratified_split.split(indices, targets):
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)

    # Apply transforms to the datasets
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']

    # Create DataLoaders
    batch_size = 32
    num_workers = 4  # Adjust based on your system
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    # Load the pretrained EfficientNet model
    model_ft = EfficientNet.from_pretrained('efficientnet-b1')
    num_ftrs = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_ftrs, num_classes)

    # Move the model to the appropriate device
    model_ft = model_ft.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=1e-4)

    # Define the learning rate scheduler
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Train the model
    num_epochs = 25  # Adjust as needed
    model_ft, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs
    )

    # Save the best model (already saved during training)
    model_path = os.path.join('best_model.pth')
    print(f'Model saved to {model_path}')

    # Plot training and validation loss
    epochs = range(1, num_epochs + 1)

    plt.figure()
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()

    # Plot training and validation accuracy
    plt.figure()
    plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_curve.png')
    plt.show()

    # Prepare the test dataset and DataLoader
    test_dataset = TestDataset(test_dir, transform=data_transforms['test'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Set the model to evaluation mode
    model_ft.eval()

    # Map class indices to class names
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Make predictions on the test data
    all_predictions = []
    all_image_paths = []

    with torch.no_grad():
        for inputs, paths in test_loader:
            inputs = inputs.to(device)
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            all_predictions.extend(preds.cpu().numpy())
            all_image_paths.extend(paths)

    # Print predictions
    for img_path, pred in zip(all_image_paths, all_predictions):
        class_name = idx_to_class[pred]
        print(f'{img_path}: {class_name}')

    # If test labels are available, compute evaluation metrics
    # Assuming test images are organized in subdirectories similar to training data
    if os.path.isdir(os.path.join(test_dir, class_names[0])):
        test_dataset_labeled = datasets.ImageFolder(root=test_dir, transform=data_transforms['test'])
        test_loader_labeled = DataLoader(test_dataset_labeled, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Make predictions on the labeled test data
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader_labeled:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model_ft(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute confusion matrix and classification report
        cm = confusion_matrix(all_labels, all_preds)
        cr = classification_report(all_labels, all_preds, target_names=class_names)

        print('Confusion Matrix:')
        print(cm)
        print('Classification Report:')
        print(cr)

        # Plot confusion matrix
        def plot_confusion_matrix(cm, class_names):
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.savefig('confusion_matrix.png')
            plt.show()

        plot_confusion_matrix(cm, class_names)

