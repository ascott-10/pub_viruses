#codes/classifier_setup.py

#### Libraries ####
import os
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import cv2

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

#### Custom ####

from config import *


#### Functions ####

def custom_train_test_split(input_images_df, save_dir=SAM_2_WORKING_DATA_DIR, stratify=True):
    """This function will take a df of all segmented images  with columns ,im_id,file_path,class and return a train/test/val split
    Usage: X_train_df, X_test_df, X_val_df = custom_train_test_split(input_images_df, save_dir=WORKING_DATA_CLASSIFIER, stratify=True)"""

    save_path_train = os.path.join(save_dir, "X_train.csv")
    save_path_test = os.path.join(save_dir, "X_test.csv")
    save_path_val  = os.path.join(save_dir, "X_val.csv")

    if os.path.exists(save_path_train):
        X_train_df = pd.read_csv(save_path_train)
        X_test_df  = pd.read_csv(save_path_test)
        X_val_df   = pd.read_csv(save_path_val)
    else:
        stratify_labels = input_images_df["class"] if stratify else None

        X_train_df, X_test_df = train_test_split(
            input_images_df,
            test_size=0.2,
            stratify=stratify_labels,
            random_state=42,
        )

        X_train_df, X_val_df = train_test_split(
            X_train_df,
            test_size=0.25,
            stratify=stratify_labels.loc[X_train_df.index] if stratify else None,
            random_state=42,
        )

        # shuffle each split
        X_train_df = X_train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        X_test_df  = X_test_df.sample(frac=1, random_state=42).reset_index(drop=True)
        X_val_df   = X_val_df.sample(frac=1, random_state=42).reset_index(drop=True)

        

        X_train_df.to_csv(save_path_train, index=False)
        X_test_df.to_csv(save_path_test, index=False)
        X_val_df.to_csv(save_path_val, index=False)
    
    print(X_train_df["class"].value_counts())
    print(X_val_df["class"].value_counts())
    print(X_test_df["class"].value_counts())


    return X_train_df, X_test_df, X_val_df


def transform_data(
    image_size=(256, 256),
    normalize_mean=(0.5,), 
    normalize_std=(0.5,),
    rotation_degree=15,
    scale_range=(0.9, 1.0),
    apply_augmentation=True
):
    base_transform = [
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ]

    if apply_augmentation:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(rotation_degree),
            transforms.RandomResizedCrop(image_size, scale=scale_range),
            *base_transform
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            *base_transform
        ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        *base_transform
    ])

    return train_transform, val_transform

def create_tensor_dataset(df, transform):
    images, labels = [], []

    for _, row in df.iterrows():
        file_path = row["file_path"]
        label_str = row["class"].lower()

        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Image not found: {file_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = transform(img)

        if label_str not in CLASS_TO_IDX:
            raise ValueError(f"Unknown class label: {label_str}")

        images.append(img)
        labels.append(CLASS_TO_IDX[label_str])

    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return TensorDataset(images_tensor, labels_tensor)

def create_dataloader(dataset, batch_size=64, shuffle=True):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

