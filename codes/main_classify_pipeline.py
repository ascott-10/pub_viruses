# codes/main_classifier_pipeline.py

#### Libraries ####
import os
import pandas as pd
import torch
from datetime import datetime

from config import *
from codes.classifier_setup import custom_train_test_split, transform_data, create_tensor_dataset, create_dataloader
from codes.classifier_trainer import load_classifier, train_model, load_resnet_weights, make_predictions

print('Imported libraries')


def run(FORCE_TRAIN_CLASSIFIER: bool = False):
    """Main classification pipeline."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Set GPU or CPU ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 0) Load segmented images
    df = pd.read_csv(SAM_2_SEGMENTED_IMAGES)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    input_images_df = df.copy()

    # 1) Train/test/val split
    X_train_df, X_test_df, X_val_df = custom_train_test_split(
        input_images_df, save_dir=WORKING_DATA_CLASSIFIER, stratify=True
    )

    # 2) Transforms
    train_transform, val_transform = transform_data(
        image_size=(256, 256),
        normalize_mean=(0.5,), 
        normalize_std=(0.5,),
        rotation_degree=15,
        scale_range=(0.9, 1.0),
        apply_augmentation=True
    )

    # 3) Create datasets/loaders
    train_dataset = create_tensor_dataset(X_train_df, train_transform)
    val_dataset = create_tensor_dataset(X_val_df, val_transform)
    test_dataset = create_tensor_dataset(X_test_df, val_transform)

    train_loader = create_dataloader(train_dataset, batch_size=64, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=64, shuffle=False)
    test_loader = create_dataloader(test_dataset, batch_size=32, shuffle=False)

    for imgs, labels in train_loader:
        print(labels[:20])
        break

    # 4) Load model
    model = load_classifier(device, num_classes=2)
    model.to(device)

    # 5) Train or load existing weights
    if FORCE_TRAIN_CLASSIFIER:
        print("FORCE_TRAIN_CLASSIFIER=True → training model from scratch...")
        train_model(model, device, train_loader, val_loader, save_dir=WORKING_DATA_CLASSIFIER)
        model = load_resnet_weights(model, device, num_classes=2, weights_path=CLASSIFIER_WEIGHTS_PATH)
    else:
        try:
            model = load_resnet_weights(model, device, num_classes=2, weights_path=CLASSIFIER_WEIGHTS_PATH)
            print("Loaded existing weights, skipping training.")
        except FileNotFoundError:
            print("No weights found, training new model...")
            train_model(model, device, train_loader, val_loader, save_dir=WORKING_DATA_CLASSIFIER)
            model = load_resnet_weights(model, device, num_classes=2, weights_path=CLASSIFIER_WEIGHTS_PATH)

    # 6) Test
    class_names = ["mut", "wt"]
    accuracy, cm, results_df = make_predictions(
        model, device, test_loader,
        save_cm=True,
        save_dir=CLASSIFIER_TESTING_RESULTS
    )

    results_path = os.path.join(CLASSIFIER_TESTING_RESULTS, f"test_predictions_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved per-image predictions → {results_path}")
