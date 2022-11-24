import matplotlib.pyplot as plt
import transforms3d.quaternions as tfq
import numpy as np
from network import ImuEventModel
from datasets import EventDataset
from torch.utils.data import DataLoader
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

delta = 10
event_bins = 1
data_dir = "indoor_forward_9_davis_with_gt"
vio_dataset = EventDataset(data_dir, delta, event_bins)

transformer_out_features = 32
model = ImuEventModel(transformer_out_features, event_bins)
model.load_state_dict(torch.load("data/event_model_0.010282640582092873.pt"))
model.to(device)
model.eval()

path = []
ground = []
x = np.zeros((3, 1))
gtx = np.zeros((3, 1))

# revert back the labels with this
q0 = vio_dataset.labels_df.loc[0, "qw":"qz"].to_numpy()

means = 0.14081615209579468
stds = 0.35239288210868835

for i in range(int(len(vio_dataset)/delta)):
    data_point = vio_dataset.__getitem__(i*delta)

    delta_label = data_point["label"].numpy().reshape(7, 1)[:3]
    rotated = tfq.rotate_vector((delta_label * stds + means).reshape((3,)), q0)
    gtx += rotated.reshape((3, 1))
    ground.append(gtx.copy())

    inputs = (data_point["events"].unsqueeze(0).to(device),
              data_point["imu"].unsqueeze(0).to(device))
    outputs = model(inputs)
    dx = outputs.cpu().detach().numpy().T[:3]
    xpred_rot = tfq.rotate_vector((dx * stds + means).reshape((3,)), q0)
    x += xpred_rot.reshape((3, 1))
    path.append(x.copy())

path = np.array(path).reshape(-1, 3)
ground = np.array(ground).reshape(-1, 3)

fig = plt.figure(dpi=200)
ax = plt.axes(projection='3d')
ax.plot3D(path[:, 0], path[:, 1], path[:, 2])
ax.plot3D(ground[:, 0], ground[:, 1], ground[:, 2])
plt.savefig("3Deval.png")

plt.figure(2, dpi=150)
plt.plot(path[:, 0], path[:, 1])
plt.plot(ground[:, 0], ground[:, 1])
plt.legend(["pred", "ground"])
plt.savefig("eval.png")
