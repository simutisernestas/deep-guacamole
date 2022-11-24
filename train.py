#!/opt/conda/bin/python
from network import ImuEventModel
from datasets import EventDataset, CombinedDataset
from torch.utils.data import DataLoader
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

delta = 10
event_bins = 1
data_dir = "indoor_forward_9_davis_with_gt"
LABEL_NORM_MEAN = 0.0 #0.14
LABEL_NORM_STD = 1.0 #0.35
vio_dataset_1 = EventDataset(
    data_dir, delta, event_bins,
    mean=LABEL_NORM_MEAN, std=LABEL_NORM_STD)
# dataloader_1 = DataLoader(vio_dataset_1, batch_size=100, shuffle=True, num_workers=4)
data_dir = "indoor_forward_3_davis_with_gt"
vio_dataset_2 = EventDataset(
    data_dir, delta, event_bins,
    mean=LABEL_NORM_MEAN, std=LABEL_NORM_STD)
# dataloader_2 = DataLoader(vio_dataset_2, batch_size=100, shuffle=True, num_workers=4)
combined_dataset = CombinedDataset([vio_dataset_1, vio_dataset_2])
combined_dataloader = DataLoader(
    combined_dataset, batch_size=100, shuffle=True, num_workers=6)

transformer_out_features = 32
model = ImuEventModel(transformer_out_features, event_bins)
CHECKPOINT = ""
if CHECKPOINT != "":
    model.load_state_dict(torch.load(CHECKPOINT))
model.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
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

        epoch_loss = running_loss / (len(vio_dataset_1)+len(vio_dataset_2))
        print(f'{phase} Loss: {epoch_loss:.8f}')

        if epoch_loss < best_epoch_loss:
            torch.save(model.state_dict(), f"data/event_model_{epoch}.pt")
            best_epoch_loss = epoch_loss

    scheduler.step()
