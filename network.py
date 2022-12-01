from torch import nn
import torch
from vit_pytorch import ViT

class IMUTransformer(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=6, nhead=6, dim_feedforward=32, batch_first=True)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.encoder = transformer_encoder
        self.fc1 = nn.Linear(in_features=6, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=out_features)

    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = torch.relu(self.fc1(x[:,-1,:]))
        x = self.fc2(x)
        return x

class ImuEventModel(torch.nn.Module):

    def __init__(self, out_features=32, event_bins=5):
        super(ImuEventModel, self).__init__()
        self.event_tf = ViT(
            image_size = 350,
            patch_size = 25,
            num_classes = out_features,
            dim = 32,
            depth = 6,
            heads = 8,
            mlp_dim = 64,
            dropout = 0.1,
            emb_dropout = 0.0,
            channels=event_bins
        )
        imu_out = int(out_features/2)
        self.imu_tf = IMUTransformer(imu_out)
        ll_dim = imu_out + out_features
        self.linear1 = torch.nn.Linear(ll_dim, ll_dim)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(ll_dim, 7)

    def forward(self, x):
        # let's say x is tuple
        events = x[0]
        imu = x[1]
        events_out = self.event_tf(events)
        imu_out = self.imu_tf(imu)
        x = torch.concat((events_out, imu_out),dim=1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
