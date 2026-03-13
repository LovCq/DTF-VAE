from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, DistributedSampler
from CVAE import CVAE, TimeCVAE
from dataset import UniDataset
import argparse
from torch import optim
from collections import OrderedDict
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import data_augment
from get_f1_score import best_f1, delay_f1, best_f1_without_pointadjust
from Attention import EncoderLayer_selfattn
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

class MyVAE(LightningModule):
    def __init__(self, hparams):
        super(MyVAE, self).__init__()
        self.save_hyperparameters()
        self.hp = hparams
        self._init_feature_dims = False
        self.num_time_features = None
        self.num_stats_features = None
        self.automatic_optimization = False
        self.warmup_epochs = 5
        self.anomaly_weight = 5.0

    def setup(self, stage=None):
        if not self._init_feature_dims:
            if stage == 'fit' or stage is None:
                train_loader = self.train_dataloader()
                sample_batch = next(iter(train_loader))
            elif stage == 'test':
                test_loader = self.test_dataloader()
                sample_batch = next(iter(test_loader))
            else:
                raise ValueError(f"Unsupported stage: {stage}")
            self.num_time_features = sample_batch['time_feats'].shape[1]
            self.num_stats_features = sample_batch['stats_feats'].shape[1]
            self.__build_model()
            self._init_feature_dims = True

    def __build_model(self):
        self.freq_vae = CVAE(self.hp, num_time_features=self.num_time_features)
        self.time_vae = TimeCVAE(self.hp, num_stats_features=self.num_stats_features)
        self.fusion_gate = EncoderLayer_selfattn(
            d_model=self.hp.latent_dim,
            d_inner=self.hp.d_inner,
            n_head=self.hp.n_head,
            dropout=0.1
        )
        self.fusion_gate_norm = nn.LayerNorm(self.hp.latent_dim)
        freq_cond_size = 2 * self.hp.condition_emb_dim
        time_cond_size = 3 * self.hp.condition_emb_dim
        total_cond_size = freq_cond_size + time_cond_size
        self.fusion_condition = nn.Linear(total_cond_size, 2 * self.hp.condition_emb_dim)
        self.fusion_decoder = nn.Sequential(
            nn.Linear(self.hp.latent_dim + 2 * self.hp.condition_emb_dim + 2, 256),
            nn.ReLU(),
            nn.Linear(256, self.hp.num_channels * self.hp.window)
        )
        self.fusion_var_x = nn.Sequential(
            nn.Linear(self.hp.num_channels * self.hp.window, self.hp.num_channels * self.hp.window),
            nn.Softplus()
        )
        self.fusion_projection = nn.Linear(2 * self.hp.latent_dim, self.hp.latent_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(2 * self.hp.latent_dim, self.hp.latent_dim),
            nn.Sigmoid()
        )
        self.fusion_attention = nn.MultiheadAttention(embed_dim=self.hp.latent_dim, num_heads=self.hp.n_head)
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.hp.condition_emb_dim, num_heads=self.hp.n_head)
        self.freq_cond_proj = nn.Linear(2 * self.hp.condition_emb_dim, self.hp.condition_emb_dim)
        self.time_cond_proj = nn.Linear(3 * self.hp.condition_emb_dim, self.hp.condition_emb_dim)
        self.fused_cond_proj = nn.Linear(self.hp.condition_emb_dim, 2 * self.hp.condition_emb_dim)

    def forward(self, x, mode, mask):
        if mode in ["train", "valid"]:
            #print(f"train_x shape: {x.shape}")
            freq_output = self.freq_vae(x, mode, mask)
            time_output = self.time_vae(x, mode, mask)
            freq_mu_x, freq_var_x, freq_rec, freq_mu, freq_var, freq_cond, freq_loss = freq_output
            time_mu_x, time_var_x, time_rec, time_mu, time_var, time_cond, time_loss = time_output
            freq_z = freq_mu
            time_z = time_mu
            freq_cond_proj = self.freq_cond_proj(freq_cond)
            time_cond_proj = self.time_cond_proj(time_cond)
            freq_cond_attn = freq_cond_proj.unsqueeze(0)
            time_cond_attn = time_cond_proj.unsqueeze(0)
            fused_cond, _ = self.cross_attention(freq_cond_attn, time_cond_attn, time_cond_attn)
            fused_cond = fused_cond.squeeze(0)
            fused_cond_proj = self.fused_cond_proj(fused_cond)
            gate_input = torch.cat([freq_z, time_z], dim=1)
            gate_weight = self.gate_net(gate_input)
            fused_z_gate = gate_weight * freq_z + (1 - gate_weight) * time_z
            freq_z_attn = freq_z.unsqueeze(0)
            time_z_attn = time_z.unsqueeze(0)
            fused_z_attn, _ = self.fusion_attention(freq_z_attn, time_z_attn, time_z_attn)
            fused_z_attn = fused_z_attn.squeeze(0)
            fused_z = fused_z_gate + fused_z_attn
            fused_z = self.fusion_gate_norm(fused_z)
            freq_corr = self.freq_vae.get_global_corr(x)
            time_corr = self.time_vae.get_global_corr(x)
            corr_feature = torch.stack([freq_corr, time_corr], dim=1)
            decoder_input = torch.cat([fused_z, fused_cond_proj, corr_feature], dim=1)
            mu_x_flat = self.fusion_decoder(decoder_input)
            mu_x = mu_x_flat.view(-1, self.hp.num_channels, self.hp.window)
            var_x_flat = self.fusion_var_x(mu_x_flat)
            var_x = var_x_flat.view(-1, self.hp.num_channels, self.hp.window)
            rec_x = self.freq_vae.reparameterize(mu_x, var_x)
            freq_var_clipped = torch.clamp(freq_var, min=1e-6, max=1e2)
            time_var_clipped = torch.clamp(time_var, min=1e-6, max=1e2)
            return [mu_x, var_x, rec_x, (freq_mu, time_mu), (freq_var_clipped, time_var_clipped),
                    (freq_loss, time_loss), freq_loss, time_loss]
        else:
            # Step 1: 子模型 MCMC 修复
            x_freq_repaired, freq_score = self.freq_vae.MCMC2(x, mask)
            x_time_repaired, time_score = self.time_vae.MCMC2(x, mask)

            # 调试：打印修复后的数据形状
            print(f"test_x shape: {x.shape}")
            print(f"x_freq_repaired shape: {x_freq_repaired.shape}")
            print(f"x_time_repaired shape: {x_time_repaired.shape}")

            # Step 2: 融合修复后的 x
            # 确保 x_freq_repaired 和 x_time_repaired 是正确的形状用于 get_conditon
            freq_cond = self.freq_vae.get_conditon(x_freq_repaired)  # No unsqueeze here
            time_cond = self.time_vae.get_conditon(
                x_time_repaired.unsqueeze(1))  # Unsqueeze here for time model if needed

            # 调试：打印条件向量形状
            print(f"freq_cond shape: {freq_cond.shape}")
            print(f"time_cond shape: {time_cond.shape}")

            # Step 3: 编码潜在变量
            freq_z = \
            self.freq_vae.encode(torch.cat([x_freq_repaired.flatten(1), freq_cond.view(x.size(0), -1)], dim=1))[0]
            time_z = self.time_vae.encode(x_time_repaired.flatten(1), time_cond.view(x.size(0), -1))[0]

            # 调试：打印潜在变量形状
            print(f"freq_z shape: {freq_z.shape}")
            print(f"time_z shape: {time_z.shape}")

            # Step 4: 融合模块
            gate_weight = self.gate_net(torch.cat([freq_z, time_z], dim=1))
            fused_z = gate_weight * freq_z + (1 - gate_weight) * time_z
            fused_z_attn, _ = self.fusion_attention(freq_z.unsqueeze(0), time_z.unsqueeze(0), time_z.unsqueeze(0))
            fused_z = self.fusion_gate_norm(fused_z + fused_z_attn.squeeze(0))

            # 调试：打印融合后的张量形状
            print(f"fused_z shape: {fused_z.shape}")

            # Step 5: 条件融合
            freq_cond_proj = self.freq_cond_proj(freq_cond.view(x.size(0), -1))
            time_cond_proj = self.time_cond_proj(time_cond.view(x.size(0), -1))

            # 调试：打印投影后的条件向量形状
            print(f"freq_cond_proj shape: {freq_cond_proj.shape}")
            print(f"time_cond_proj shape: {time_cond_proj.shape}")

            fused_cond, _ = self.cross_attention(freq_cond_proj.unsqueeze(0), time_cond_proj.unsqueeze(0),
                                                 time_cond_proj.unsqueeze(0))
            fused_cond = self.fused_cond_proj(fused_cond.squeeze(0))

            # 调试：打印融合后的条件向量形状
            print(f"fused_cond shape: {fused_cond.shape}")

            # Step 6: 解码
            corr_feature = torch.stack(
                [self.freq_vae.get_global_corr(x_freq_repaired), self.time_vae.get_global_corr(x_time_repaired)], dim=1)

            # 调试：打印全局相关特征形状
            print(f"corr_feature shape: {corr_feature.shape}")

            decoder_input = torch.cat([fused_z, fused_cond, corr_feature], dim=1)
            mu_x_flat = self.fusion_decoder(decoder_input)
            mu_x_fused = mu_x_flat.view(-1, self.hp.num_channels, self.hp.window)

            # 调试：打印解码后的输出形状
            print(f"mu_x_fused shape: {mu_x_fused.shape}")

            # Step 7: 计算异常分数（通过重构误差和概率计算）
            var_x_est = torch.ones_like(mu_x_fused) * 1.0  # 简化
            fused_prob = -0.5 * (torch.log(var_x_est + 1e-8) + (x - mu_x_fused) ** 2 / (var_x_est + 1e-8))
            fused_score = fused_prob.mean(dim=1)  # 计算异常分数

            # 调试：打印最终的异常分数
            print(f"fused_score shape: {fused_score.shape}")

            # Step 8: 计算 final_score
            freq_score = freq_score.squeeze(1)  # 确保 freq_score 维度为 [batch_size, window]
            #time_score = time_score.squeeze(1)  # 确保 time_score 维度为 [batch_size, window]
            print(
                f"freq_score shape: {freq_score.shape}, time_score shape: {time_score.shape}, fused_score shape: {fused_score.shape}")
            final_score = freq_score
            #final_score = 0.75 * freq_score + 0.3 * time_score + 0.7 * fused_score  # 最终异常分数

            return mu_x_fused, final_score


    def loss(self, x, y_all, z_all, mode="train"):
        mask = torch.logical_not(torch.logical_or(y_all, z_all))
        mu_x, var_x, rec_x, (freq_mu, time_mu), (freq_var, time_var), (
            total_freq_loss, total_time_loss), freq_loss, time_loss = self.forward(x, mode, mask)
        temperature = 1.0
        freq_weight = F.softmax(torch.tensor([freq_loss / temperature], device=x.device), dim=0)[0]
        time_weight = F.softmax(torch.tensor([time_loss / temperature], device=x.device), dim=0)[0]
        if self.hp.use_contrast and mode == "train":
            normal_mask = (y_all == 0).all(dim=1)
            anomaly_mask = ~normal_mask
            if normal_mask.sum() > 1 and anomaly_mask.sum() > 0:
                normal_z = torch.cat([freq_mu[normal_mask], time_mu[normal_mask]], dim=1)
                anomaly_z = torch.cat([freq_mu[anomaly_mask], time_mu[anomaly_mask]], dim=1)
                anchor = normal_z
                positive = normal_z.roll(1, dims=0)
                num_normal = normal_z.size(0)
                num_anomaly = anomaly_z.size(0)
                if num_anomaly < num_normal:
                    indices = torch.randint(0, num_anomaly, (num_normal,), device=anomaly_z.device)
                else:
                    indices = torch.randperm(num_anomaly, device=anomaly_z.device)[:num_normal]
                negative = anomaly_z[indices]
                triplet_loss = F.triplet_margin_loss(anchor, positive, negative, margin=1.0)
            else:
                triplet_loss = torch.tensor(0.0, device=x.device)
        else:
            triplet_loss = torch.tensor(0.0, device=x.device)
        recon_loss = F.huber_loss(mu_x, x, delta=1.0, reduction='none').mean(dim=[1, 2]).mean()
        kl_weight = min(1.0, self.current_epoch / self.warmup_epochs) * self.hp.kld_weight
        kld_freq = -0.5 * torch.sum(1 + freq_var - freq_mu.pow(2) - freq_var.exp(), dim=1).mean()
        kld_time = -0.5 * torch.sum(1 + time_var - time_mu.pow(2) - time_var.exp(), dim=1).mean()
        corr_loss = self.corr_loss(x, y_all)
        total_loss = (
                freq_weight * freq_loss +
                time_weight * time_loss +
                recon_loss +
                self.hp.contrast_weight * triplet_loss +
                self.hp.corr_weight * corr_loss +
                kl_weight * (kld_freq + kld_time)
        )
        return total_loss

    def corr_loss(self, x, y_all):
        freq_corr = self.freq_vae.get_global_corr(x)
        time_corr = self.time_vae.get_global_corr(x)
        anomaly_ratio = y_all[:, -1].float().mean()
        corr_loss = torch.mean(y_all[:, -1] * (freq_corr + time_corr)) * (1.0 - anomaly_ratio)
        return corr_loss

    def training_step(self, data_batch, batch_idx):
        x = data_batch['x']
        y_all = data_batch['y']
        z_all = data_batch['z']
        x, y_all, z_all = self.batch_data_augmentation(x, y_all, z_all)
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        optimizer.zero_grad()
        loss_val = self.loss(x, y_all, z_all)
        self.manual_backward(loss_val)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
        optimizer.step()
        if self.trainer.strategy == "dp":
            loss_val = loss_val.unsqueeze(0)
        self.log("val_loss_train", loss_val, on_step=True, on_epoch=False, logger=True)
        return OrderedDict({"loss": loss_val})

    def validation_step(self, data_batch, batch_idx):
        x = data_batch['x']
        y_all = data_batch['y']
        z_all = data_batch['z']
        mask = torch.logical_not(torch.logical_or(y_all, z_all))
        mu_x, var_x, rec_x, (freq_mu, time_mu), (freq_var, time_var), (total_freq_loss, total_time_loss), \
            freq_loss, time_loss = self.forward(x, "valid", mask)
        loss_val = self.loss(x, y_all, z_all, mode="valid")
        corr_loss = self.corr_loss(x, y_all)
        if self.trainer.strategy == "dp":
            loss_val = loss_val.unsqueeze(0)
            corr_loss = corr_loss.unsqueeze(0)
        self.log("val_loss_valid", loss_val, on_step=True, on_epoch=True, logger=True)
        self.log("corr_loss_valid", corr_loss, on_step=True, on_epoch=True, logger=True)
        return {"loss": loss_val, "corr_loss": corr_loss}

    def on_validation_epoch_end(self):
        val_loss_epoch = self.trainer.callback_metrics.get("val_loss_valid")
        if val_loss_epoch is not None:
            scheduler = self.lr_schedulers()
            if isinstance(scheduler, list):
                scheduler[0].step(val_loss_epoch)
            else:
                scheduler.step(val_loss_epoch)
        freq_loss_epoch = self.trainer.callback_metrics.get("freq_loss_valid")
        time_loss_epoch = self.trainer.callback_metrics.get("time_loss_valid")
        if freq_loss_epoch is not None and time_loss_epoch is not None:
            ratio = time_loss_epoch / (freq_loss_epoch + 1e-8)
            if ratio > 1.2:
                self.hp.alpha = max(0.1, self.hp.alpha - 0.01)
            elif ratio < 0.8:
                self.hp.alpha = min(0.9, self.hp.alpha + 0.01)
            self.log("alpha", self.hp.alpha, on_epoch=True)

    def on_test_epoch_start(self):
        self.test_outputs = []
        #self.test_metrics = {}

    def test_step(self, data_batch, batch_idx):
        x, y_all, z_all = data_batch['x'], data_batch['y'], data_batch['z']
        y = y_all[:, -1].unsqueeze(1)  # Last timestep label
        mask = torch.logical_not(torch.logical_or(y_all, z_all))
        with torch.no_grad():
            freq_mu_x, freq_prob = self.freq_vae(x, "test", mask)  # [B, C, W]
            time_mu_x, time_prob = self.time_vae(x, "test", mask)  # [B, W]
            freq_var = self.freq_vae.fc_var_x(freq_mu_x).mean(dim=1)  # [B, W]
            time_var = self.time_vae.fc_var_x(time_mu_x.unsqueeze(1)).mean(dim=1)  # [B, W]
            weight_freq = 1.0 / (freq_var + 1e-8)  # [B, W]
            weight_time = 1.0 / (time_var + 1e-8)  # [B, W]
            weight_sum = weight_freq + weight_time
            weight_freq = torch.clamp(weight_freq / weight_sum, min=0.3, max=0.7)
            weight_time = torch.clamp(weight_time / weight_sum, min=0.3, max=0.7)

            time_mu_x_expanded = time_mu_x.unsqueeze(1).expand(-1, self.hp.num_channels, -1)  # [B, C, W]
            time_prob_expanded = time_prob.unsqueeze(1).expand(-1, self.hp.num_channels, -1)  # [B, C, W]

            # =========================
            # 新增：动态融合模块推理（最小改动版本）
            # =========================
            # 注意：这里 freq_mu_x / time_mu_x 是 mode="test" 返回的 repaired x
            x_freq_repaired = freq_mu_x  # [B, C, W]
            x_time_repaired = time_mu_x  # [B, W]

            # 1) 用 repaired x 走 valid 路径拿 latent/cond（动态融合的输入对齐 MCMC）
            freq_out = self.freq_vae(x_freq_repaired, "valid", mask)
            # time 模型如果 valid 输入需要 [B,1,W]，就 unsqueeze(1)
            time_in = x_time_repaired.unsqueeze(1)  # [B,1,W]
            time_out = self.time_vae(time_in, "valid", mask)

            freq_mu = freq_out[3]
            freq_cond = freq_out[5]
            time_mu = time_out[3]
            time_cond = time_out[5]

            # 2) condition 动态融合（与训练一致）
            freq_cond_proj = self.freq_cond_proj(freq_cond)
            time_cond_proj = self.time_cond_proj(time_cond)
            fused_cond, _ = self.cross_attention(
                freq_cond_proj.unsqueeze(0),
                time_cond_proj.unsqueeze(0),
                time_cond_proj.unsqueeze(0),
            )
            fused_cond_proj = self.fused_cond_proj(fused_cond.squeeze(0))

            # 3) latent 动态融合（与训练一致）
            gate_weight = self.gate_net(torch.cat([freq_mu, time_mu], dim=1))
            fused_z_gate = gate_weight * freq_mu + (1.0 - gate_weight) * time_mu

            fused_z_attn, _ = self.fusion_attention(
                freq_mu.unsqueeze(0),
                time_mu.unsqueeze(0),
                time_mu.unsqueeze(0),
            )
            fused_z = self.fusion_gate_norm(fused_z_gate + fused_z_attn.squeeze(0))

            # 4) corr_feature：建议仍用原始 x（与训练一致），或用 repaired x（二选一）
            freq_corr = self.freq_vae.get_global_corr(x)  # 或 x_freq_repaired
            time_corr = self.time_vae.get_global_corr(x)  # 或 x_time_repaired.unsqueeze(1)
            corr_feature = torch.stack([freq_corr, time_corr], dim=1)

            # 5) fusion_decoder 输出 mu_x_test
            decoder_input = torch.cat([fused_z, fused_cond_proj, corr_feature], dim=1)
            mu_x_flat = self.fusion_decoder(decoder_input)
            mu_x_test = mu_x_flat.view(-1, self.hp.num_channels, self.hp.window)

            # 6) recon_prob：先保留基线的 MCMC 概率加权（不要换 fusion var 版本）
            # recon_prob 本来就是你前面算出来的静态融合概率项：
            recon_prob = weight_freq.unsqueeze(1) * freq_prob + weight_time.unsqueeze(1) * time_prob_expanded

            # 7) anomaly_score：仍沿用你原来的公式（最小改动）
            recon_error = torch.abs(x - mu_x_test)
            anomaly_score = recon_error * (1 - recon_prob)
            anomaly_score = anomaly_score.mean(dim=1)  # [B, W]

            x_last = x[:, 0, -1]

        outputs = OrderedDict({
            "y": y.cpu().flatten(),
            "anomaly_score": anomaly_score.cpu(),  # [batch_size, window]
            "mu_x_test": mu_x_test.cpu(),  # [batch_size, num_channels, window]
            "x": x_last.cpu().flatten(),
            "x_full": x.cpu(),
            "y_full": y_all.cpu(),
        })
        self.test_outputs.append(outputs)
        return outputs

    # def test_step(self, data_batch, batch_idx):
    #     x, y_all, z_all = data_batch['x'], data_batch['y'], data_batch['z']
    #     y = y_all[:, -1].unsqueeze(1)  # Last timestep label
    #     mask = torch.logical_not(torch.logical_or(y_all, z_all))
    #
    #     with torch.no_grad():
    #         # Use forward's test mode
    #         mu_x_fused, final_score = self.forward(x, mode="test", mask=mask)
    #
    #         # Return outputs with updated keys
    #         outputs = OrderedDict({
    #             "y": y.cpu().flatten(),
    #             "anomaly_score": final_score.cpu(),  # Using final_score from forward
    #             "mu_x_test": mu_x_fused.cpu(),  # [batch_size, num_channels, window]
    #             "x": x[:, 0, -1].cpu().flatten(),  # Last timestep of first channel
    #             "x_full": x.cpu(),
    #             "y_full": y_all.cpu(),
    #         })
    #         self.test_outputs.append(outputs)
    #         return outputs

    def on_test_epoch_end(self):
        all_y = torch.cat([output['y'] for output in self.test_outputs], dim=0).numpy()
        all_anomaly_score = torch.cat([output['anomaly_score'] for output in self.test_outputs], dim=0).numpy()  # [n_samples, window]
        all_mu_x_test = torch.cat([output['mu_x_test'] for output in self.test_outputs], dim=0).numpy()
        all_x = torch.cat([output['x'] for output in self.test_outputs], dim=0).numpy()
        # all_x_full = torch.cat([output['x_full'] for output in self.test_outputs], dim=0).numpy()
        # all_y_full = torch.cat([output['y_full'] for output in self.test_outputs], dim=0).numpy()
        # all_freq_mu_x = torch.cat([output['freq_mu_x'] for output in self.test_outputs], dim=0).numpy()
        # all_time_mu_x = torch.cat([output['time_mu_x'] for output in self.test_outputs], dim=0).numpy()

        df = pd.DataFrame({
            'y': all_y,
            'anomaly_score': all_anomaly_score[:, -1],  # Last timestep for consistency
            'mu_x_test': all_mu_x_test[:, 0, -1],
            'x': all_x
        })
        score = df["anomaly_score"].values
        label = df["y"].values
        np.save("./npy/score.npy", score)
        np.save("./npy/label.npy", label)
        if self.hp.data_dir == "./data/Yahoo":
            k = 3
        elif self.hp.data_dir == "./data/NAB" or self.hp.data_dir == "./data/new_NAB":
            k = 150
        else:
            k = 50
        try:
            auc = roc_auc_score(y_true=label, y_score=score)
        except ValueError:
            if len(np.unique(label)) == 1:
                print("WARNING: All labels are same class. AUC undefined.")
                auc = 0.5
            else:
                raise
        delay_f1_score, delay_precision, delay_recall, delay_predict = delay_f1(score, label, k)
        best_f1_score, best_precision, best_recall, best_predict = best_f1(score, label)
        best_f1_score_, best_precision_, best_recall_, best_predict_ = best_f1_without_pointadjust(score, label)
        df["delay_predict"] = delay_predict
        df["best_predict"] = best_predict
        df.to_csv("./csv/result.csv", index=False)
        file_name = self.hp.save_file
        directory = os.path.dirname(file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_name, "a") as f:
            f.write(
                "Auc %f \nbest f1 score %f %f %f \nDelay f1 score  %f %f %f\nBest f1 without pointadjust %f %f %f\n"
                % (
                    auc,
                    best_f1_score,
                    best_precision,
                    best_recall,
                    delay_f1_score,
                    delay_precision,
                    delay_recall,
                    best_f1_score_,
                    best_precision_,
                    best_recall_,
                )
            )
        self.log_dict({
            "Metrics/AUC": auc,
            "Metrics/Best_F1": best_f1_score,
            "Metrics/Delay_F1": delay_f1_score
        }, on_epoch=True)
        #metrics = {
        #    "Metrics/AUC": auc,
        #    "Metrics/Best_F1": best_f1_score,
        #    "Metrics/Delay_F1": delay_f1_score
        #}
        #self.log_dict(metrics, on_epoch=True)
        #self.test_metrics = metrics
        self.logger.experiment.add_scalar("Metrics/Best_F1", best_f1_score, self.current_epoch)
        self.logger.experiment.add_scalar("Metrics/Best_Precision", best_precision, self.current_epoch)
        self.logger.experiment.add_scalar("Metrics/Best_Recall", best_recall, self.current_epoch)
        self.logger.experiment.add_scalar("Metrics/Delay_F1", delay_f1_score, self.current_epoch)
        self.logger.experiment.add_scalar("Metrics/Delay_Precision", delay_precision, self.current_epoch)
        self.logger.experiment.add_scalar("Metrics/Delay_Recall", delay_recall, self.current_epoch)
        #self.test_metrics = metrics

        self.test_outputs = []

    def train_dataloader(self):
        return self.mydataloader('train')

    def val_dataloader(self):
        return self.mydataloader('valid')

    def test_dataloader(self):
        return self.mydataloader('test')

    def mydataloader(self, mode):
        dataset = UniDataset(
            self.hp.use_label,
            self.hp.window,
            self.hp.data_dir,
            self.hp.data_name,
            mode,
            self.hp.sliding_window_size,
            data_pre_mode=self.hp.data_pre_mode,
        )
        # ===== model.py / MyVAE.mydataloader 里：dataset 创建后加入 =====
        if not hasattr(self, "_printed_dataset_stats"):
            self._printed_dataset_stats = set()

        if mode not in self._printed_dataset_stats:
            self._printed_dataset_stats.add(mode)

            print("\n" + "=" * 80)
            print(f"[Dataset Stats] mode={mode}")
            print(f"  data_dir: {self.hp.data_dir}")
            print(f"  window(W): {self.hp.window}, sliding_window_size: {self.hp.sliding_window_size}")
            print(f"  #files: {getattr(dataset, 'n_files', 'NA')}")
            print(f"  raw_rows_after_split (sum over files): {getattr(dataset, 'total_raw_rows', 'NA')}")
            print(f"  completed_len (sum): {getattr(dataset, 'total_completed_len', 'NA')}")
            print(f"  smoothed_len (sum): {getattr(dataset, 'total_smoothed_len', 'NA')}")
            print(f"  window_samples (=len(dataset)): {len(dataset)}")
            # 可选：也打印 total_windows 对照
            if hasattr(dataset, "total_windows"):
                print(f"  total_windows (check): {dataset.total_windows}")

            # 可选：打印前几个文件明细
            if hasattr(dataset, "file_stats") and len(dataset.file_stats) > 0:
                print("  per-file (first 3):")
                for d in dataset.file_stats[:3]:
                    print(f"    {d['file']}: raw={d['raw_rows_after_split']}, "
                          f"completed={d['completed_len']}, smoothed={d['smoothed_len']}, windows={d['window_samples']}")
            print("=" * 80 + "\n")

        num_channels = dataset.samples.shape[1]
        self.hp.num_channels = num_channels
        train_sampler = None
        batch_size = self.hp.batch_size
        try:
            if self.on_gpu:
                train_sampler = DistributedSampler(dataset, rank=self.trainer.proc_rank)
                batch_size = batch_size // self.trainer.world_size
        except Exception:
            pass
        should_shuffle = train_sampler is None
        if mode in ["valid", "test"]:
            should_shuffle = False
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=0,
        )
        return loader

    def configure_optimizers(self):
        optimizer = optim.Adam([
            {'params': self.freq_vae.parameters(), 'lr': self.hp.learning_rate * 0.5, 'weight_decay': 1e-4},
            {'params': self.time_vae.parameters(), 'lr': self.hp.learning_rate, 'weight_decay': 1e-4},
            {'params': (
                    list(self.fusion_gate.parameters()) +
                    list(self.fusion_gate_norm.parameters()) +
                    list(self.fusion_condition.parameters()) +
                    list(self.fusion_decoder.parameters()) +
                    list(self.fusion_var_x.parameters()) +
                    list(self.fusion_projection.parameters()) +
                    list(self.gate_net.parameters()) +
                    list(self.fusion_attention.parameters()) +
                    list(self.cross_attention.parameters())
            ), 'lr': self.hp.learning_rate * 0.8, 'weight_decay': 1e-4}
        ])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_name", default=None, type=str)
        parser.add_argument("--ckpt_path", default=None, type=str)
        parser.add_argument("--data_dir", default="./data/WSD/", type=str)
        parser.add_argument("--window", default=128, type=int)
        parser.add_argument("--latent_dim", default=8, type=int)
        parser.add_argument("--only_test", default=0, type=int)
        parser.add_argument("--max_epoch", default=30, type=int)
        parser.add_argument("--batch_size", default=512, type=int)
        parser.add_argument("--num_workers", default=0, type=int)
        parser.add_argument("--learning_rate", default=0.0005, type=float)
        parser.add_argument("--sliding_window_size", default=1, type=int)
        parser.add_argument("--save_file", default="./result/Score.txt", type=str)
        parser.add_argument("--data_pre_mode", default=0, type=int)
        parser.add_argument("--missing_data_rate", default=0.01, type=float)
        parser.add_argument("--point_ano_rate", default=0.02, type=float)
        parser.add_argument("--seg_ano_rate", default=0.05, type=float)
        parser.add_argument("--eval_all", default=0, type=int)
        parser.add_argument("--condition_emb_dim", default=16, type=int)
        parser.add_argument("--d_model", default=256, type=int)
        parser.add_argument("--d_inner", default=512, type=int)
        parser.add_argument("--n_head", default=8, type=int)
        parser.add_argument("--kernel_size", default=8, type=int)
        parser.add_argument("--stride", default=8, type=int)
        parser.add_argument("--mcmc_rate", default=0.2, type=float)
        parser.add_argument("--mcmc_value", default=-5, type=float)
        parser.add_argument("--mcmc_mode", default=2, type=int)
        parser.add_argument("--num_channels", default=1, type=int)
        parser.add_argument("--condition_mode", default=2, type=int)
        parser.add_argument("--dropout_rate", default=0, type=float)
        parser.add_argument("--gpu", default=0, type=int)
        parser.add_argument("--use_label", default=0, type=int)
        parser.add_argument("--alpha", type=float, default=0.6)
        parser.add_argument("--use_contrast", type=int, default=1)
        parser.add_argument("--contrast_weight", type=float, default=0.2)
        parser.add_argument("--fusion_dim", type=int, default=128)
        parser.add_argument("--corr_weight", type=float, default=0.5)
        parser.add_argument("--kld_weight", type=float, default=0.005)
        return parser

    def batch_data_augmentation(self, x, y, z):
        device = x.device
        point_ano_rate = self.hp.point_ano_rate * (1 - self.current_epoch / self.trainer.max_epochs)
        seg_ano_rate = self.hp.seg_ano_rate * (1 - self.current_epoch / self.trainer.max_epochs)
        if point_ano_rate > 0:
            x_a, y_a, z_a = data_augment.point_ano(x, y, z, point_ano_rate)
            x = torch.cat((x, x_a.to(device)), dim=0)
            y = torch.cat((y, y_a.to(device)), dim=0)
            z = torch.cat((z, z_a.to(device)), dim=0)
        if seg_ano_rate > 0:
            x_a, y_a, z_a = data_augment.seg_ano(x, y, z, seg_ano_rate, method="swap")
            x = torch.cat((x, x_a.to(device)), dim=0)
            y = torch.cat((y, y_a.to(device)), dim=0)
            z = torch.cat((z, z_a.to(device)), dim=0)
        x, y, z = data_augment.missing_data_injection(x, y, z, self.hp.missing_data_rate * 0.5)
        return x.to(device), y.to(device), z.to(device)