from constants import *
import torch.nn.functional as F
import transforms3d.quaternions as tfq
import transforms3d as tf
import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
import math
from constants import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EventDataset(Dataset):
    def __init__(self, data_dir, label_delta_len, event_bins, mean=0.0, std=1.0):
        imu_cols = ["timestamp", "ang_vel_x", "ang_vel_y",
                    "ang_vel_z", "lin_acc_x", "lin_acc_y", "lin_acc_z"]
        self.imu_df = pd.read_csv(os.path.join(
            data_dir, "imu.txt"), delimiter=' ', skiprows=1, names=imu_cols)
        self.imu_df.index = pd.to_datetime(self.imu_df["timestamp"], unit='s')
        imu_df_nostamp = self.imu_df.loc[:, self.imu_df.columns != 'timestamp']
        self.imu_data = imu_df_nostamp.to_numpy(dtype=np.float32)
        self.imu_data = torch.from_numpy(self.imu_data)

        events_cols = ["timestamp", "x", "y", "polarity"]
        events_df = pd.read_csv(os.path.join(
            data_dir, "events.txt"), delimiter=' ', skiprows=1, names=events_cols)
        events_df_nostamp = events_df.loc[:, events_df.columns != 'timestamp']
        self.events_data = torch.from_numpy(
            events_df_nostamp.to_numpy(dtype=np.float32)).type(torch.LongTensor)

        indexed_labels_pickle = f"data/{data_dir}_{label_delta_len}.pickle"
        if os.path.isfile(indexed_labels_pickle):
            self.labels_df = pd.read_pickle(indexed_labels_pickle)
            self.labels_df["timestamp"] = pd.to_datetime(
                self.labels_df["timestamp"], unit='s')
        else:
            label_cols = ["timestamp", "tx", "ty",
                          "tz", "qx", "qy", "qz", "qw", "none"]
            labels_df = pd.read_csv(os.path.join(
                data_dir, "groundtruth.txt"), delimiter=' ', skiprows=1, names=label_cols)
            labels_df = labels_df.iloc[:, :-1]
            # rearange quaternion, w goes first
            columns_titles = ["timestamp", "tx",
                              "ty", "tz", "qw", "qx", "qy", "qz"]
            labels_df = labels_df.reindex(columns=columns_titles)
            labels_df["events_start_idx"] = 0
            self.labels_df = labels_df
            labels_df_nostamp = labels_df.loc[:,
                                              labels_df.columns != 'timestamp']
            labels_df_nostamp = labels_df_nostamp.loc[:,
                                                      labels_df_nostamp.columns != 'events_start_idx']
            labels = labels_df_nostamp.to_numpy()

            q0 = labels[0, 3:]  # take first one and align everything to it
            q0inv = tfq.qinverse(q0)
            # q0, q0inv, tfq.qmult(q0,q0inv), tfq.nearly_equivalent(tfq.qmult(q0,q0inv),tfq.qeye())

            new_cols = ["dtx", "dty", "dtz", "dqw", "dqx", "dqy", "dqz"]
            for col in new_cols:
                self.labels_df[col] = 0.0
            first_id = abs(events_df["timestamp"] -
                           self.labels_df.iloc[0]["timestamp"]).idxmin()
            for i in range(len(self.labels_df)-label_delta_len):
                label_ts = self.labels_df.iloc[i, 0]
                # hopefully no more than 10k event between labels
                id = events_df[first_id:first_id +
                               int(1e4)]["timestamp"].sub(label_ts).abs().idxmin()
                self.labels_df["events_start_idx"][i] = id
                first_id = id
                if i % 1000 == 0 and i != 0:
                    print(i)

                dpose = np.zeros(7, dtype=np.float64)
                l1 = labels[i]
                l2 = labels[i+label_delta_len]
                dl = tfq.rotate_vector(
                    l2[:3], q0inv) - tfq.rotate_vector(l1[:3], q0inv)  # position delta
                dpose[:3] = dl
                q1 = tfq.qmult(l1[3:], q0inv)  # [[3,0,1,2]]
                q2 = tfq.qmult(l2[3:], q0inv)  # [[3,0,1,2]]
                # q_b = q_diff * q_a
                # q_b * inv(q_a) = q_diff * q_a * inv(q_a)
                # q_b * inv(q_a) = q_diff
                dq = tf.quaternions.qmult(
                    q2, tf.quaternions.qinverse(q1))  # orientation delta
                dpose[3:] = dq  # pose delta is the label

                # dpose = np.zeros(7, dtype=np.float64)
                # l1 = labels[i]
                # l2 = labels[i+label_delta_len]
                # dl = l2[:3] - l1[:3]
                # dpose[:3] = dl
                # q1 = l1[3:]  # [[3,0,1,2]]
                # q2 = l2[3:]  # [[3,0,1,2]]
                # dq = tf.quaternions.qmult(
                #     q1, tf.quaternions.qinverse(q2))  # orientation delta
                # dpose[3:] = dq  # pose delta is the label

                for col_index, col in enumerate(new_cols):
                    self.labels_df[col][i+label_delta_len] = dpose[col_index]

            # save for easy loading
            self.labels_df.to_pickle(indexed_labels_pickle)
            self.labels_df["timestamp"] = pd.to_datetime(
                self.labels_df["timestamp"], unit='s')

        self.labels = torch.tensor(
            self.labels_df.loc[:, 'dtx':'dqz'].to_numpy(), dtype=torch.float32)
        print(f"Loaded: {data_dir}")
        # print(f"mean = {mean}\nstds = {std}")
        self.labels = (self.labels - mean) / std

        self.delta = label_delta_len
        self.bins = event_bins

    def __len__(self):
        actual_len = math.floor((self.labels.shape[0] - self.delta))
        return actual_len #- (actual_len % BATCH_SIZE)  # / self.delta

    def __getitem__(self, idx):
        start_idx = idx  # * self.delta
        end_idx = start_idx + self.delta

        dpose = self.labels[end_idx]

        start = self.labels_df.iloc[start_idx]["events_start_idx"]
        finish = self.labels_df.iloc[end_idx]["events_start_idx"]

        if self.bins > 1:
            num_per_bin = math.floor((finish-start)/self.bins)
            binned_events = torch.zeros((self.bins, mc, mc))
            for b in range(self.bins):
                start_bin = int(start)+b*num_per_bin
                if b != (self.bins - 1):
                    events = self.events_data[start_bin:start_bin+num_per_bin]
                else:
                    events = self.events_data[start_bin:int(finish)]
                binned_events[b] = torch.index_put(binned_events[b], (events[:, 1], events[:, 0]), (
                    events[:, 2].type(torch.FloatTensor) - .5), accumulate=True)
        else:
            binned_events = torch.zeros((self.bins, mc, mc))
            events = self.events_data[int(start):int(finish)]
            binned_events[0] = torch.index_put(binned_events[0], (events[:, 1], events[:, 0]), (
                events[:, 2].type(torch.FloatTensor) - .5), accumulate=True)

        imu_index_start = self.imu_df.index.get_loc(
            self.labels_df.iloc[start_idx]["timestamp"], method='nearest')
        imu_index_end = self.imu_df.index.get_loc(
            self.labels_df.iloc[end_idx]["timestamp"], method='nearest')
        imu = self.imu_data[imu_index_start:imu_index_end]

        p1d = (0, 0, 65-imu.shape[0], 0)
        imu = F.pad(imu, p1d, "constant", 0)

        sample = {"events": binned_events, "imu": imu, "label": dpose}

        return sample

    def set_label_delta(self, delta):
        self.delta = delta


class CombinedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, index):
        ret = []
        for dataset in self.datasets:
            ret.append(dataset[index % len(dataset)])
        return tuple(ret)

    def __len__(self):
        return max([len(x) for x in self.datasets])
