import argparse
import os, sys, time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from datasets import XRaysTrainDataset, XRaysTestDataset
from trainer import fit
import config
from sklearn.metrics import roc_auc_score

# Handle device configuration
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'\nDevice: {device}')

# Argument parser
parser = argparse.ArgumentParser(description='Arguments for model training/testing.')
parser.add_argument('--data_path', type=str, default='NIH Chest X-rays', help='Path to the training data')
parser.add_argument('--bs', type=int, default=128, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--stage', type=int, default=1, help='Stage of training')
parser.add_argument('--loss_func', type=str, default='FocalLoss', choices={'BCE', 'FocalLoss'}, help='Loss function')
parser.add_argument('-r', '--resume', action='store_true', help='Resume training')
parser.add_argument('--ckpt', type=str, help='Checkpoint path to load')
parser.add_argument('-t', '--test', action='store_true', help='Run in test mode')
args = parser.parse_args()

if args.resume and args.test:
    sys.exit('Cannot resume training and test simultaneously. Choose one.')

stage = args.stage
if not args.resume:
    print(f'\nOverwriting stage to 1 as training is being done from scratch')
    stage = 1

if args.test:
    print('TESTING THE MODEL')
else:
    print('RESUMING TRAINING' if args.resume else 'TRAINING FROM SCRATCH')

# Timer for script runtime
script_start_time = time.time()

# Dataset configuration
data_dir = args.data_path
XRayTrain_dataset = XRaysTrainDataset(data_dir, transform=config.transform)
train_percentage = 0.8
train_dataset, val_dataset = torch.utils.data.random_split(
    XRayTrain_dataset, [int(len(XRayTrain_dataset) * train_percentage),
                        len(XRayTrain_dataset) - int(len(XRayTrain_dataset) * train_percentage)]
)
XRayTest_dataset = XRaysTestDataset(data_dir, transform=config.transform)

# Dataloaders
batch_size = args.bs
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(XRayTest_dataset, batch_size=batch_size, shuffle=False)

# Loss function with class imbalance handling
class_counts = [100, 200, 300, 50, 80, 70, 120, 40, 60, 90, 110, 130, 45, 55, 95]  # Replace with real counts
total_samples = sum(class_counts)
class_weights = [total_samples / (len(class_counts) * count) for count in class_counts]
class_weights_tensor = torch.tensor(class_weights).to(device)

if args.loss_func == 'FocalLoss':
    from losses import FocalLoss
    loss_fn = FocalLoss(device=device, gamma=2.0).to(device)
elif args.loss_func == 'BCE':
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor).to(device)

# Learning rate and scheduler
lr = args.lr
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim.Adam(filter(lambda p: p.requires_grad, models.resnet50(pretrained=True).parameters()), lr=lr),
    mode='min', factor=0.1, patience=3, verbose=True
)

# Mixed precision training setup
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()

if not args.test:
    # Initialize or resume model
    if not args.resume:
        print('\nTraining from scratch')
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(XRayTrain_dataset.all_classes))
        model.to(device)
        epochs_till_now = 0
        losses_dict = {'epoch_train_loss': [], 'epoch_val_loss': [], 'total_train_loss_list': [], 'total_val_loss_list': []}
    else:
        if not args.ckpt:
            sys.exit('ERROR: Please provide a valid checkpoint to resume from')
        ckpt = torch.load(os.path.join(config.models_dir, args.ckpt))
        epochs_till_now = ckpt['epochs']
        model = ckpt['model']
        model.to(device)
        losses_dict = ckpt['losses_dict']

    # Stage-specific configurations
    if stage == 1:
        for name, param in model.named_parameters():
            param.requires_grad = ('layer2' in name or 'layer3' in name or 'layer4' in name or 'fc' in name)
    elif stage == 2:
        for name, param in model.named_parameters():
            param.requires_grad = ('layer3' in name or 'layer4' in name or 'fc' in name)
    elif stage == 3:
        for name, param in model.named_parameters():
            param.requires_grad = ('layer4' in name or 'fc' in name)
    elif stage == 4:
        for name, param in model.named_parameters():
            param.requires_grad = ('fc' in name)
    elif stage == 5:  # Fine-tuning stage
        print('\n----- STAGE 5 -----')
        for name, param in model.named_parameters():
            param.requires_grad = True  # Fine-tune all layers

    # Optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)

    # Training
    fit(device, XRayTrain_dataset, train_loader, val_loader, test_loader, model, loss_fn, optimizer,
        losses_dict, epochs_till_now=epochs_till_now, epochs=10, log_interval=25, save_interval=1, lr=lr,
        bs=batch_size, stage=stage, test_only=args.test, scheduler=scheduler)

if args.test:
    # Testing
    if not args.ckpt:
        sys.exit('ERROR: Please provide a checkpoint to load the testing model from')
    ckpt = torch.load(os.path.join(config.models_dir, args.ckpt))
    model = ckpt['model']
    model.eval()  # Set the model to evaluation mode

    y_true, y_probs = [], []
    for batch in test_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            probabilities = torch.sigmoid(outputs)  # Get probabilities
        y_true.append(labels.cpu().numpy())
        y_probs.append(probabilities.cpu().numpy())

    y_true = np.vstack(y_true)  # Shape: [n_samples, n_classes]
    y_probs = np.vstack(y_probs)  # Shape: [n_samples, n_classes]

    # Compute ROC-AUC for each class and macro-average
    roc_auc_per_class = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:  # Avoid single-class issues
            roc_auc = roc_auc_score(y_true[:, i], y_probs[:, i])
            roc_auc_per_class.append(roc_auc)
            print(f"Class {i} ({XRayTest_dataset.all_classes[i]}): ROC-AUC = {roc_auc:.4f}")
        else:
            roc_auc_per_class.append(None)  # Undefined ROC-AUC for single-class cases
            print(f"Class {i} ({XRayTest_dataset.all_classes[i]}): ROC-AUC = N/A (only one class present)")

    # Calculate the macro-average ROC-AUC, ignoring classes with undefined ROC-AUC
    roc_auc_macro = np.mean([auc for auc in roc_auc_per_class if auc is not None])
    print(f"\nOverall Metrics:\nMacro-Average ROC-AUC: {roc_auc_macro:.4f}")

script_time = time.time() - script_start_time
print(f"Script completed in {int(script_time // 3600)}h {int((script_time % 3600) // 60)}m {int(script_time % 60)}s!")



"""
This code is adapted from the original implementation:
'NIH Chest X-Rays Multi-Label Image Classification in PyTorch'

Original Author: n0obcoder
Original Repository: https://github.com/n0obcoder/NIH-Chest-X-Rays-Multi-Label-Image-Classification-In-Pytorch
Reference: 
    @misc{n0obcoder2023github,
      author = {n0obcoder},
      year = {2023},
      title = {NIH Chest X-Rays Multi-Label Image Classification in PyTorch},
      howpublished = {\href{https://github.com/n0obcoder/NIH-Chest-X-Rays-Multi-Label-Image-Classification-In-Pytorch}{GitHub Repository}},
      note = {Accessed: 20/12/2024}
    }

Credit goes to the original author for their excellent work and foundational implementation.
This adaptation includes modifications for extended functionality and additional training stages.
"""