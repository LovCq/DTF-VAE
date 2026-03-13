# import torch
# import torch.utils.data # PyTorch 数据加载工具
# import logging
# import numpy as np
# import pandas as pd
# import os
# import datapreprocess # 自定义预处理模块
#
#
# class UniDataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         use_label,
#         window,
#         data_dir,
#         data_name,
#         mode,
#         sliding_window_size,
#         data_pre_mode=0,
#     ):
#         self.window = window
#         self.data_dir = data_dir
#         self.data_name = data_name
#         file_list = os.listdir(data_dir)
#         value_all = []
#         label_all = []
#         missing_all = []
#         self.sample_num = 0
#
#         for file in file_list:
#             file_path = os.path.join(data_dir, file)
#             df = pd.read_csv(file_path)
#             df_train = df[: int(0.35 * len(df))]
#             df_train = df_train.bfill()
#
#             # 假设多变量数据列以 "value" 开头，例如 "value1", "value2", ...
#             value_columns = [col for col in df.columns if col.startswith("value")]
#             num_channels = len(value_columns)
#             train_values = np.asarray(df_train[value_columns])  # [样本数, num_channels]
#             train_label = np.asarray(df_train["label"])
#             train_values_normal = train_values[np.where(train_label == 0)[0]]  # 仅用正常点计算
#             train_max = train_values_normal.max(axis=0)  # 每通道最大值
#             train_min = train_values_normal.min(axis=0)  # 每通道最小值
#
#             if mode == "train":
#                 df = df[: int(0.35 * len(df))]
#             elif mode == "valid":
#                 df = df[int(0.35 * len(df)): int(0.5 * len(df))]
#             elif mode == "test":
#                 df = df[int(0.5 * len(df)):]
#
#             timestamp, missing, (values, label) = datapreprocess.complete_timestamp(
#                 df["timestamp"], (df[value_columns].values, df["label"])
#             )
#             values = values.astype(float)  # [时间步长, num_channels]
#             missing2 = np.any(np.isnan(values), axis=1)  # 检测任一通道缺失
#             missing = np.logical_or(missing, missing2).astype(int)
#             label = label.astype(float)
#             label[np.where(missing == 1)[0]] = np.nan
#             values[np.where(missing == 1)[0]] = np.nan
#
#             df2 = pd.DataFrame({"timestamp": timestamp, "label": label, "missing": missing})
#             for i, col in enumerate(value_columns):
#                 df2[col] = values[:, i]
#             df2 = df2.bfill().fillna(0)
#             df2["label"] = df2["label"].astype(int)
#
#             if data_pre_mode == 0:
#                 for i, col in enumerate(value_columns):
#                     df2[col], *_ = datapreprocess.standardize_kpi(df2[col])
#             else:
#                 for i, col in enumerate(value_columns):
#                     v = np.asarray(df2[col])
#                     v = 2 * (v - train_min[i]) / (train_max[i] - train_min[i] + 1e-8) - 1
#                     df2[col] = v
#
#             timestamp = np.asarray(df2["timestamp"])
#             values = np.clip(np.asarray(df2[value_columns]), -40, 40)  # [时间步长, num_channels]
#             labels = np.asarray(df2["label"])
#             missing = np.asarray(df2["missing"])
#             values[np.where(missing == 1)[0]] = 0
#
#             if (mode == "train" or mode == "valid") and use_label == 1:
#                 values[np.where(labels == 1)[0]] = 0
#             elif (mode == "train" or mode == "valid") and use_label == 0:
#                 labels[:] = 0
#
#             # 对每通道应用滑动平均
#             smoothed_values = np.zeros((len(values) - sliding_window_size + 1, num_channels))
#             for ch in range(num_channels):
#                 smoothed_values[:, ch] = np.convolve(
#                     values[:, ch],
#                     np.ones(sliding_window_size) / sliding_window_size,
#                     mode="valid",
#                 )
#             timestamp = timestamp[sliding_window_size - 1:]
#             labels = labels[sliding_window_size - 1:]
#             missing = missing[sliding_window_size - 1:]
#
#             value_all.append(smoothed_values)
#             label_all.append(labels)
#             missing_all.append(missing)
#             self.sample_num += max(len(smoothed_values) - window + 1, 0)
#
#         self.num_channels = num_channels
#         self.samples, self.labels, self.miss_label = self.__getsamples(value_all, label_all, missing_all)
#         self.time_features = self._extract_time_features()
#         self.stats_features = self._calc_window_stats()
#
#     def __getsamples(self, values, labels, missing):
#         X = torch.zeros((self.sample_num, self.num_channels, self.window))
#         Y = torch.zeros((self.sample_num, self.window))
#         Z = torch.zeros((self.sample_num, self.window))
#         i = 0
#         for cnt in range(len(values)):
#             v = values[cnt]  # [时间步长, num_channels]
#             l = labels[cnt]
#             m = missing[cnt]
#             for j in range(len(v) - self.window + 1):
#                 X[i, :, :] = torch.from_numpy(v[j: j + self.window].T)  # [num_channels, window]
#                 Y[i, :] = torch.from_numpy(l[j: j + self.window])
#                 Z[i, :] = torch.from_numpy(m[j: j + self.window])
#                 i += 1
#         return X, Y, Z
#
#     def __len__(self):
#         return self.sample_num  # 返回总样本数
#
#
#     def _extract_time_features(self):
#         # 示例：假设时间戳已保存为self.timestamps
#         if not hasattr(self, 'timestamps'):
#             return torch.zeros((self.sample_num, 3))  # 默认3个特征：小时、周几、月
#         timestamps = pd.to_datetime(self.timestamps)
#         features = np.stack([
#             timestamps.hour.values,
#             timestamps.dayofweek.values,
#             timestamps.month.values
#         ], axis=1)
#         return torch.FloatTensor(features[:self.sample_num])
#
#     def _calc_window_stats(self):
#         windows = self.samples  # [sample_num, num_channels, window]
#         means = windows.mean(dim=2)  # [sample_num, num_channels]
#         stds = windows.std(dim=2)  # [sample_num, num_channels]
#         return torch.cat([means, stds], dim=1)  # [sample_num, num_channels * 2]
#
#     def __getitem__(self, idx):
#         return {
#             'x': self.samples[idx],
#             'y': self.labels[idx],
#             'z': self.miss_label[idx],  # 添加缺失标记
#             'time_feats': self.time_features[idx],
#             'stats_feats': self.stats_features[idx]
#         }
import torch
import torch.utils.data  # PyTorch 数据加载工具
import logging
import numpy as np
import pandas as pd
import os
import datapreprocess  # 自定义预处理模块


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

        # ===== 新增：统计字段 =====
        self.n_files = 0
        self.total_raw_rows = 0            # 该 mode 下，切分后原始 df 行数总和（切分前未补全时间戳）
        self.total_completed_len = 0       # complete_timestamp 后的长度（补齐后）
        self.total_smoothed_len = 0        # 平滑后长度（len(smoothed_values)）
        self.total_windows = 0             # 窗口化后的样本数（应与 self.sample_num 一致）
        self.file_stats = []               # 每个文件的明细统计（可选）

        # 如果目录为空，num_channels 后面会用到，这里先兜底
        num_channels = None

        for file in file_list:
            file_path = os.path.join(data_dir, file)
            df_full = pd.read_csv(file_path)

            # 假设多变量数据列以 "value" 开头，例如 "value1", "value2", ...
            value_columns = [col for col in df_full.columns if col.startswith("value")]
            num_channels = len(value_columns)
            if num_channels == 0:
                raise ValueError(f"No value columns found in {file_path}. Expect columns start with 'value'.")

            # ========= 先按整文件算 train_min/train_max（与你原逻辑一致：只用训练段，且仅正常点计算） =========
            df_train = df_full[: int(0.35 * len(df_full))]
            df_train = df_train.bfill()

            train_values = np.asarray(df_train[value_columns])  # [样本数, num_channels]
            train_label = np.asarray(df_train["label"])
            train_values_normal = train_values[np.where(train_label == 0)[0]]  # 仅用正常点计算
            # 注意：若训练段全是异常点，train_values_normal 可能为空，这里做个保护
            if train_values_normal.shape[0] == 0:
                # 退化处理：用全训练段计算 min/max（避免报错）
                train_max = train_values.max(axis=0)
                train_min = train_values.min(axis=0)
            else:
                train_max = train_values_normal.max(axis=0)  # 每通道最大值
                train_min = train_values_normal.min(axis=0)  # 每通道最小值

            # ========= 计算切分边界（0.35 / 0.15 / 0.50） =========
            n_full = len(df_full)
            n_train = int(0.35 * n_full)
            n_valid_end = int(0.5 * n_full)  # 0.35 + 0.15

            # ========= 根据 mode 做切片（与你现在一致） =========
            if mode == "train":
                df = df_full[:n_train]
            elif mode == "valid":
                df = df_full[n_train:n_valid_end]
            elif mode == "test":
                df = df_full[n_valid_end:]
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # ===== 新增：累计切分后原始行数 =====
            self.n_files += 1
            self.total_raw_rows += len(df)

            # ========= 你的原有 complete_timestamp / missing / bfill / 标准化... =========
            timestamp, missing, (values, label) = datapreprocess.complete_timestamp(
                df["timestamp"], (df[value_columns].values, df["label"])
            )

            # ===== 新增：累计补齐后长度 =====
            self.total_completed_len += len(timestamp)

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

            # ========= 对每通道应用滑动平均（固定窗口大小） =========
            # 注意：当 len(values) < sliding_window_size 时，这里会出现负维度，做保护
            if len(values) - sliding_window_size + 1 <= 0:
                smoothed_values = np.zeros((0, num_channels), dtype=np.float32)
            else:
                smoothed_values = np.zeros((len(values) - sliding_window_size + 1, num_channels))
                for ch in range(num_channels):
                    smoothed_values[:, ch] = np.convolve(
                        values[:, ch],
                        np.ones(sliding_window_size) / sliding_window_size,
                        mode="valid",
                    )

            # 同步裁剪 timestamp / labels / missing
            if len(timestamp) >= sliding_window_size:
                timestamp = timestamp[sliding_window_size - 1:]
                labels = labels[sliding_window_size - 1:]
                missing = missing[sliding_window_size - 1:]
            else:
                timestamp = timestamp[:0]
                labels = labels[:0]
                missing = missing[:0]

            # ===== 新增：累计平滑后长度 与 窗口数 =====
            sm_len = len(smoothed_values)
            win_cnt = max(sm_len - window + 1, 0)
            self.total_smoothed_len += sm_len
            self.total_windows += win_cnt

            # 可选：记录每个文件明细
            self.file_stats.append({
                "file": file,
                "raw_rows_after_split": len(df),
                "completed_len": int(len(timestamp)),
                "smoothed_len": int(sm_len),
                "window_samples": int(win_cnt),
            })

            value_all.append(smoothed_values)
            label_all.append(labels)
            missing_all.append(missing)

            # 你原来的累计保持不变（这里直接用 win_cnt 更清晰）
            self.sample_num += win_cnt

        if num_channels is None:
            raise ValueError(f"No files found under data_dir={data_dir}, cannot build dataset.")

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
        # 你原代码没有真正维护 self.timestamps，这里保持原行为：若没有就返回 0 特征
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
