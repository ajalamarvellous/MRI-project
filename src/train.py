import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import mlflow
from models import ConvNet
from prepare_dataset import DicomDataset, stratify_split
from trainer import Trainer
from utils import load_model, return_ylabels, seedall

# PARAMETERS
seedall(42)
lr = 0.01
epochs = 100
batch_size = 4
num_workers = 0
num_classes = 1
val_size = 0.3
test_size = 0.2
device = torch.device("cpu")
fresh_training = True
last_stop = 10
MODEL_ARTIFACT = "../models/batch_4/ConvNet_9.pth"
FILE_PATH = "../data/training.csv"
MLFLOW_TRACKING_URI = "../mlflow/"

# setting mlflow parameters
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("MRI Hearing ache project")
# getting dataset
dataset = DicomDataset(FILE_PATH)
y_labels = return_ylabels(FILE_PATH)
# Get samplers
splits, class_weight = stratify_split(y_labels, test_size=test_size)
train_sampler = SubsetRandomSampler(indices=splits[1])
test_sampler = SubsetRandomSampler(indices=splits[0])

# for batch_size in batch_size_:
mlflow.log_param("Eval param", "bEST")
# Get dataloaders
train_loader = DataLoader(
    dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
)
test_loader = DataLoader(
    dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
)


# Defining Model
model = ConvNet(num_classes)
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

if not fresh_training:
    checkpoint = load_model(MODEL_ARTIFACT)
    epochs = epochs - last_stop
    model.load_state_dict(checkpoint["model_checkpoint"])
    optimizer.load_state_dict(checkpoint["optimiser_checkpoint"])

# Specifying optimizer and criterion
criterion = torch.nn.BCELoss(weight=torch.Tensor([class_weight]))
trainer = Trainer(
    epochs=epochs,
    lr=lr,
    model=model,
    criterion=criterion,
    optimiser=optimizer,
    device=device,
    track_mlflow=True,
    model_dir="../models/new_1",
    result_dir="../results/new_1",
)
# training
trainer.train(train_loader, test_loader, eval_every=1)
