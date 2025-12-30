# codes/classifier_trainer.py

#### Libraries ####
import os
from datetime import datetime
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models

from config import *
from utils import load_images


#### Functions ####
def load_classifier(device, num_classes=2):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    model.to(device)
    return model


def plot_loss_and_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, save_dir=CLASSIFIER_TRAINING_RESULTS):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, color=TRAIN_COLOR, label='Train Loss')
    plt.plot(epochs, val_losses, color=VAL_COLOR, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, color=TRAIN_COLOR, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, color=VAL_COLOR, label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "loss_accuracy_curve.png")
        plt.savefig(save_path)
        print(f"Loss/accuracy plot saved at {save_path}")

    plt.show(block=False)
    plt.pause(2)
    plt.close()


########################################################################    
def train(model, device, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS_CLASSIFY, save_dir=WORKING_DATA_CLASSIFIER):
    model.train()

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(num_epochs):
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)
        scheduler.step()

        model.eval()
        val_correct, val_total, val_loss_total = 0, 0, 0.0

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                if val_images.shape[1] == 1:
                    val_images = val_images.repeat(1, 3, 1, 1)
                val_images, val_labels = val_images.to(device), val_labels.to(device)

                val_outputs = model(val_images)
                _, val_predicted = torch.max(val_outputs, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

                loss = criterion(val_outputs, val_labels)
                val_loss_total += loss.item()

        avg_val_loss = val_loss_total / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)

        model.train()

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}% | Validation Accuracy: {val_accuracy:.2f}%')

    print("Training complete.")
    plot_loss_and_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, save_dir=save_dir)


########################################################################    
def train_model(model, device, train_loader, val_loader, save_dir=WORKING_DATA_CLASSIFIER):
    # --- Phase 1: train only FC ---
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-6
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    train(model, device, train_loader, val_loader, criterion, optimizer, scheduler,
          num_epochs=NUM_EPOCHS_CLASSIFY, save_dir=save_dir)

    # --- Phase 2: fine-tune all layers ---
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-6
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    train(model, device, train_loader, val_loader, criterion, optimizer, scheduler,
          num_epochs=NUM_EPOCHS_CLASSIFY, save_dir=save_dir)

    # --- Save weights ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_dir = os.path.join(save_dir, "resnet_weights")
    os.makedirs(weights_dir, exist_ok=True)

    save_path = os.path.join(weights_dir, f"resnet_weights_{timestamp}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


#####################################    
def load_resnet_weights(model, device, num_classes=2, weights_path=CLASSIFIER_WEIGHTS_PATH):
    weight_files = glob.glob(os.path.join(weights_path, "resnet_weights_*.pth"))
    if not weight_files:
        raise FileNotFoundError(f"No weight files found in {weights_path}")

    if isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
    elif isinstance(model.fc, nn.Sequential):
        in_features = model.fc[-1].in_features
    else:
        raise TypeError(f"Unexpected model.fc type: {type(model.fc)}")

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )

    latest_file = max(weight_files, key=os.path.getctime)
    print(f"[INFO] Using latest weights: {latest_file}")

    state_dict = torch.load(latest_file, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


##############################################    
def make_predictions(model, device, test_loader, save_cm=True, save_dir=CLASSIFIER_TESTING_RESULTS):
    predictions, targets, file_paths = [], [], []
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels, *extras in test_loader:
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if extras:
                file_paths.extend(extras[0])

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # use config dicts
    pred_labels = [IDX_TO_CLASS[i] for i in predictions]
    true_labels = [IDX_TO_CLASS[i] for i in targets]

    results_df = pd.DataFrame({
        "true_label": true_labels,
        "predicted_label": pred_labels,
        "correct": [t == p for t, p in zip(true_labels, pred_labels)]
    })
    if file_paths:
        results_df["file_path"] = file_paths

    cm = confusion_matrix(targets, predictions, labels=list(IDX_TO_CLASS.keys()))

    fig, ax = plt.subplots(figsize=(7, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=list(IDX_TO_CLASS.values()),
        yticklabels=list(IDX_TO_CLASS.values()),
        cbar=False,
        square=True,
        linewidths=0.7,
        linecolor="gray",
        ax=ax,
        annot_kws={"size": FONT_SIZE + 6}
    )
    ax.set_xlabel("Predicted Label", fontsize=FONT_SIZE_LABEL + 4, labelpad=10)
    ax.set_ylabel("True Label", fontsize=FONT_SIZE_LABEL + 4, labelpad=10)
    ax.set_title("Confusion Matrix", fontsize=FONT_SIZE_TITLE + 6, pad=15)
    ax.tick_params(labelsize=FONT_SIZE_TICK + 6)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_cm and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cm_path = os.path.join(save_dir, f"confusion_matrix__{timestamp}.png")
        plt.savefig(cm_path, dpi=300)
        print(f"Saved confusion matrix: {cm_path}")
    plt.show()

    return accuracy, cm, results_df
