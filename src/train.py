import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from models import ConvNet
from prepare_dataset import DicomDataset, stratify_split
from trainer import Trainer
from utils import return_ylabels, seedall

# PARAMETERS
seedall(42)
lr = 0.001
epochs = 10
batch_size = 16
num_workers = 0
num_classes = 1
val_size = 0.3
test_size = 0.2
device = torch.device("cpu")
FILE_PATH = "../data/temp_data.csv"

# getting dataset
dataset = DicomDataset(FILE_PATH)
y_labels = return_ylabels(FILE_PATH)
# Get samplers
splits = stratify_split(y_labels, val_size=val_size, test_size=test_size)
train_sampler = SubsetRandomSampler(indices=splits[2])
test_sampler = SubsetRandomSampler(indices=splits[1])
val_sampler = SubsetRandomSampler(indices=splits[0])
# Get dataloaders
train_loader = DataLoader(
    dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
)
test_loader = DataLoader(
    dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
)
val_loader = DataLoader(
    dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
)
# Defining Model
model = ConvNet(num_classes)
model = model.to(device)
# Specifying optimizer and criterion
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
criterion = torch.nn.BCELoss()
trainer = Trainer(
    epochs=epochs,
    lr=lr,
    model=model,
    criterion=criterion,
    optimiser=optimizer,
    device=device,
)
# training
trainer.train(train_loader, test_loader)
