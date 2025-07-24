import torch
import torch.utils.data # PyTorch 数据加载工具
import logging
import numpy as np
import pandas as pd
import os
import datapreprocess # 自定义预处理模块


class UniDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        use_label,
        window,
        data_dir,
        data_name,
        mode,
        sliding_window_size,
        data_pre_mode=0,
    ):
        self.window = window
        self.data_dir = data_dir
        self.data_name = data_name
        file_list = os.listdir(data_dir)
        value_all = []
        label_all = []
        missing_all = []
        self.sample_num = 0

        for file in file_list:
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path)
            df_train = df[: int(0.35 * len(df))]
            df_train = df_train.bfill()

            # 假设多变量数据列以 "value" 开头，例如 "value1", "value2", ...
            value_columns = [col for col in df.columns if col.startswith("value")]
            num_channels = len(value_columns)
            train_values = np.asarray(df_train[value_columns])  # [样本数, num_channels]
            train_label = np.asarray(df_train["label"])
            train_values_normal = train_values[np.where(train_label == 0)[0]]  # 仅用正常点计算
            train_max = train_values_normal.max(axis=0)  # 每通道最大值
            train_min = train_values_normal.min(axis=0)  # 每通道最小值

            if mode == "train":
                df = df[: int(0.35 * len(df))]
            elif mode == "valid":
                df = df[int(0.35 * len(df)): int(0.5 * len(df))]
            elif mode == "test":
                df = df[int(0.5 * len(df)):]

            timestamp, missing, (values, label) = datapreprocess.complete_timestamp(
                df["timestamp"], (df[value_columns].values, df["label"])
            )
            values = values.astype(float)  # [时间步长, num_channels]
            missing2 = np.any(np.isnan(values), axis=1)  # 检测任一通道缺失
            missing = np.logical_or(missing, missing2).astype(int)
            label = label.astype(float)
            label[np.where(missing == 1)[0]] = np.nan
            values[np.where(missing == 1)[0]] = np.nan

            df2 = pd.DataFrame({"timestamp": timestamp, "label": label, "missing": missing})
            for i, col in enumerate(value_columns):
                df2[col] = values[:, i]
            df2 = df2.bfill().fillna(0)
            df2["label"] = df2["label"].astype(int)

            if data_pre_mode == 0:
                for i, col in enumerate(value_columns):
                    df2[col], *_ = datapreprocess.standardize_kpi(df2[col])
            else:
                for i, col in enumerate(value_columns):
                    v = np.asarray(df2[col])
                    v = 2 * (v - train_min[i]) / (train_max[i] - train_min[i] + 1e-8) - 1
                    df2[col] = v

            timestamp = np.asarray(df2["timestamp"])
            values = np.clip(np.asarray(df2[value_columns]), -40, 40)  # [时间步长, num_channels]
            labels = np.asarray(df2["label"])
            missing = np.asarray(df2["missing"])
            values[np.where(missing == 1)[0]] = 0

            if (mode == "train" or mode == "valid") and use_label == 1:
                values[np.where(labels == 1)[0]] = 0
            elif (mode == "train" or mode == "valid") and use_label == 0:
                labels[:] = 0

            # 对每通道应用滑动平均
            smoothed_values = np.zeros((len(values) - sliding_window_size + 1, num_channels))
            for ch in range(num_channels):
                smoothed_values[:, ch] = np.convolve(
                    values[:, ch],
                    np.ones(sliding_window_size) / sliding_window_size,
                    mode="valid",
                )
            timestamp = timestamp[sliding_window_size - 1:]
            labels = labels[sliding_window_size - 1:]
            missing = missing[sliding_window_size - 1:]

            value_all.append(smoothed_values)
            label_all.append(labels)
            missing_all.append(missing)
            self.sample_num += max(len(smoothed_values) - window + 1, 0)

        self.num_channels = num_channels
        self.samples, self.labels, self.miss_label = self.__getsamples(value_all, label_all, missing_all)
        self.time_features = self._extract_time_features()
        self.stats_features = self._calc_window_stats()

    def __getsamples(self, values, labels, missing):
        X = torch.zeros((self.sample_num, self.num_channels, self.window))
        Y = torch.zeros((self.sample_num, self.window))
        Z = torch.zeros((self.sample_num, self.window))
        i = 0
        for cnt in range(len(values)):
            v = values[cnt]  # [时间步长, num_channels]
            l = labels[cnt]
            m = missing[cnt]
            for j in range(len(v) - self.window + 1):
                X[i, :, :] = torch.from_numpy(v[j: j + self.window].T)  # [num_channels, window]
                Y[i, :] = torch.from_numpy(l[j: j + self.window])
                Z[i, :] = torch.from_numpy(m[j: j + self.window])
                i += 1
        return X, Y, Z

    def __len__(self):
        return self.sample_num  # 返回总样本数


    def _extract_time_features(self):
        # 示例：假设时间戳已保存为self.timestamps
        if not hasattr(self, 'timestamps'):
            return torch.zeros((self.sample_num, 3))  # 默认3个特征：小时、周几、月
        timestamps = pd.to_datetime(self.timestamps)
        features = np.stack([
            timestamps.hour.values,
            timestamps.dayofweek.values,
            timestamps.month.values
        ], axis=1)
        return torch.FloatTensor(features[:self.sample_num])

    def _calc_window_stats(self):
        windows = self.samples  # [sample_num, num_channels, window]
        means = windows.mean(dim=2)  # [sample_num, num_channels]
        stds = windows.std(dim=2)  # [sample_num, num_channels]
        return torch.cat([means, stds], dim=1)  # [sample_num, num_channels * 2]

    def __getitem__(self, idx):
        return {
            'x': self.samples[idx],
            'y': self.labels[idx],
            'z': self.miss_label[idx],  # 添加缺失标记
            'time_feats': self.time_features[idx],
            'stats_feats': self.stats_features[idx]
        }