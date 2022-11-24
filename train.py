#!/usr/bin/env python3
import sys
from network import ImuEventModel
from datasets import EventDataset, CombinedDataset
from torch.utils.data import DataLoader
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if len(sys.argv) != 4:
    print("Provide checkpoint and LR!")
    exit()

LABEL_NORM_MEAN = 0.0  # 0.14
LABEL_NORM_STD = 1.0  # 0.35
delta = 32  # TODO: changed : )
event_bins = 1  # TODO: THERE IS A BUG WITH MULTIPLE BINS! AT RUNTIME DATALOADER FAILS

# process directories
data_dirs = [
    "indoor_forward_9_davis_with_gt",
    "indoor_forward_3_davis_with_gt",
    "indoor_forward_7_davis_with_gt"
]
datasets = []
for data_dir in data_dirs:
    vio_dataset = EventDataset(
        data_dir, delta, event_bins,
        mean=LABEL_NORM_MEAN, std=LABEL_NORM_STD)
    datasets.append(vio_dataset)

# combine into single dataset
combined_dataset = CombinedDataset(datasets)
combined_dataloader = DataLoader(
    combined_dataset, batch_size=int(sys.argv[3]), shuffle=True, num_workers=16)

transformer_out_features = 128  # TODO: changed : )
model = ImuEventModel(transformer_out_features, event_bins)
CHECKPOINT = sys.argv[1]
if CHECKPOINT != "":
    model.load_state_dict(torch.load(CHECKPOINT))
model.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=float(sys.argv[2]))
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[100, 150], gamma=.5)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, factor=.5, patience=2, threshold=0.01)

# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html training loop from here
num_epochs = 200
best_epoch_loss = torch.inf
for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)

    for phase in ['train']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0

        for batches in combined_dataloader:
            for batch in batches:
                inputs = (batch["events"].to(device), batch["imu"].to(device))
                labels = batch["label"].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # + torch.norm(1 - torch.norm(outputs[3:]), 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * batch["imu"].size(0)

        epoch_loss = running_loss / sum([len(d) for d in datasets])
        print(f'{phase} Loss: {epoch_loss:.8f}')

        if epoch_loss < best_epoch_loss:
            torch.save(model.state_dict(), f"data/event_model.pt")
            best_epoch_loss = epoch_loss

    scheduler.step()
