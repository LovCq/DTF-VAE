import numpy as np
import torch


def missing_data_injection(x, y, z, rate):
    miss_size = int(rate * x.shape[0] * x.shape[1] * x.shape[2])
    row = torch.randint(low=0, high=x.shape[0], size=(miss_size,), device=x.device)
    chan = torch.randint(low=0, high=x.shape[1], size=(miss_size,), device=x.device)
    col = torch.randint(low=0, high=x.shape[2], size=(miss_size,), device=x.device)
    x[row, chan, col] = 0
    z[row, col] = 1
    return x, y, z


def point_ano(x, y, z, rate):
    aug_size = int(rate * x.shape[0])
    id_x = torch.randint(low=0, high=x.shape[0], size=(aug_size,), device=x.device)
    x_aug = x[id_x].clone()
    y_aug = y[id_x].clone()
    z_aug = z[id_x].clone()
    ano_noise = torch.randn(aug_size, x.shape[1], device=x.device) * 10  # 多通道噪声
    x_aug[:, :, -1] += ano_noise
    y_aug[:, -1] = torch.logical_or(y_aug[:, -1], torch.ones_like(y_aug[:, -1]))
    return x_aug, y_aug, z_aug


def seg_ano(x, y, z, rate, method):
    """
      功能：通过交换两个样本的时间段来注入段级异常。
      参数：
          x (Tensor): 输入数据，形状为 [样本数, 特征数, 时间步长]
          y (Tensor): 原始标签（将被修改）
          z (Tensor): 缺失标记
          rate (float): 增强样本比例（0~1）
          method (str): 仅支持 "swap"（交换时间段）
      返回：
          x_aug (Tensor): 增强后的数据（含异常段）
          y_aug (Tensor): 更新后的标签（标记异常段）
          z_aug (Tensor): 未变化的缺失标记
      """
    aug_size = int(rate * x.shape[0])
    # 生成索引时指定设备
    idx_1 = torch.randint(low=0, high=x.shape[0], size=(aug_size,), device=x.device)
    idx_2 = torch.randint(low=0, high=x.shape[0], size=(aug_size,), device=x.device)
    while torch.any(idx_1 == idx_2):
        idx_1 = torch.randint(low=0, high=x.shape[0], size=(aug_size,), device=x.device)
        idx_2 = torch.randint(low=0, high=x.shape[0], size=(aug_size,), device=x.device)
    x_aug = x[idx_1].clone()
    y_aug = y[idx_1].clone()
    z_aug = z[idx_1].clone()
    time_start = torch.randint(low=7, high=x.shape[2], size=(aug_size,), device=x.device)
    for i in range(len(idx_2)):
        if method == "swap":
            x_aug[i, :, time_start[i]:] = x[idx_2[i], :, time_start[i]:]
            y_aug[:, time_start[i]:] = torch.logical_or(y_aug[:, time_start[i]:],
                                                        torch.ones_like(y_aug[:, time_start[i]:]))
    return x_aug, y_aug, z_aug
