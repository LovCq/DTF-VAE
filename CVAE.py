import torch
import numpy as np
import math
from torch import nn
from torch.nn import functional as F
from Attention import EncoderLayer_selfattn
import pywt

class CVAE(nn.Module):
    def __init__(
        self,
        hp,
        num_time_features,
        gamma: float = 1000.0,
        max_capacity: int = 25,
        Capacity_max_iter: int = 1e5,
        loss_type: str = "M",
        freq_mask_ratio: float = 0.3
    ):
        super(CVAE, self).__init__()
        self.hp = hp
        self.num_time_features = num_time_features
        self.num_channels = hp.num_channels
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.freq_mask_ratio = freq_mask_ratio
        self.m_freq = nn.Parameter(torch.randn(1, 1, hp.d_model) * 0.01)
        self.num_iter = 0

        # Condition encoder
        self.condition_encoder = nn.Sequential(
            nn.Linear(num_time_features, hp.condition_emb_dim),
            nn.LayerNorm(hp.condition_emb_dim),
            nn.ReLU()
        )
        self.conv = nn.Conv1d(
            in_channels=self.num_channels,
            out_channels=hp.d_model,
            kernel_size=hp.kernel_size,
            stride=hp.stride
        )
        self._calc_conv_output_size()
        self.hidden_dims = [100, 100]
        self.encoder = nn.Sequential(
            nn.Linear(self.hp.window + 2 * self.hp.condition_emb_dim, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh()
        )
        self.fc_mu = nn.Linear(100, hp.latent_dim)
        self.fc_var = nn.Sequential(nn.Linear(100, hp.latent_dim), nn.Softplus())
        self.decoder_input = nn.Linear(
            self.hp.latent_dim + 2 * self.hp.condition_emb_dim, self.hidden_dims[-1]
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1]),
            nn.Tanh(),
            nn.Linear(self.hidden_dims[-1], hp.window)
        )
        self.fc_mu_x = nn.Linear(self.hp.window, self.hp.window)
        self.fc_var_x = nn.Sequential(
            nn.Linear(self.hp.window, self.hp.window), nn.Softplus()
        )
        self.atten = nn.ModuleList(
            [EncoderLayer_selfattn(self.hp.d_model, self.hp.d_inner, self.hp.n_head, dropout=0.1)
             for _ in range(1)]
        )
        self.emb_local = nn.Sequential(
            nn.Linear(2 + self.hp.kernel_size, self.hp.d_model),
            nn.Tanh(),
        )
        self.out_linear = nn.Sequential(
            nn.Linear(self.hp.d_model, self.hp.condition_emb_dim),
            nn.Tanh(),
        )
        self.dropout = nn.Dropout(self.hp.dropout_rate)
        self.emb_global = nn.Sequential(
            nn.Linear(hp.window, 256),
            nn.ReLU(),
            nn.Linear(256, hp.condition_emb_dim),
            nn.Tanh()
        )


    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        var = self.fc_var(result)
        return [mu, var]

    def decode(self, z, condition):
        decoder_in = torch.cat([z, condition], dim=1)
        result = self.decoder_input(decoder_in)
        result = result.view(-1, 1, self.hidden_dims[0])
        result = self.decoder(result)
        mu_x = self.fc_mu_x(result)
        var_x = self.fc_var_x(result)
        return mu_x, var_x

    def reparameterize(self, mu, var):
        std = torch.sqrt(1e-7 + var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, mode, y):
        #print(f"[DEBUG] input shape: {input.shape}, mean: {input.mean().item():.4f}, min: {input.min().item():.4f}, max: {input.max().item():.4f}")
        #print(f"[DEBUG] y shape: {y.float().shape}, mean: {y.float().mean().item():.4f}, True比例: {(y == 1).float().mean().item():.4f}, False比例: {(y == 0).float().mean().item():.4f}")
        if mode in ["train", "valid"]:
            condition = self.get_conditon(input)
            condition_flat = condition.view(input.size(0), -1)
            input_flat = input.view(input.size(0), -1)  # [batch_size, num_channels * window]
            encoder_input = torch.cat([input_flat, condition_flat], dim=1)
            mu, var = self.encode(encoder_input)
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(z, condition_flat)
            rec_x = self.reparameterize(mu_x, var_x)
            loss = self.loss_func(mu_x, var_x, input, mu, var, y, z)
            # **Modification**: Return condition_flat instead of duplicating loss
            return [mu_x, var_x, rec_x, mu, var, condition_flat, loss]
        else:
            return self.MCMC2(input, y)

    def get_conditon(self, x):
        x_g = x  # [batch_size, num_channels, window]
        f_global = torch.fft.rfft(x_g, dim=-1)  # [batch_size, num_channels, n_freq]
        real, imag = f_global.real, f_global.imag
        amplitude = torch.sqrt(real ** 2 + imag ** 2)
        batch_size, channels, n_freq = amplitude.shape
        k = int(n_freq * self.freq_mask_ratio)
        _, mask_indices = torch.topk(-amplitude.mean(dim=1), k, dim=-1)  # 沿通道平均
        masked_real = real.clone()
        masked_imag = imag.clone()
        for b in range(batch_size):
            for c in range(channels):
                masked_real[b, c, mask_indices[b]] = self.m_freq[..., 0]
                masked_imag[b, c, mask_indices[b]] = self.m_freq[..., 1]
        masked_freq = torch.complex(masked_real, masked_imag)
        masked_time = torch.fft.irfft(masked_freq, n=x.size(-1), dim=-1)
        f_global = self.emb_global(masked_time.mean(dim=1))  # 沿通道平均, [batch_size, condition_emb_dim]

        x_g = x_g.view(x.shape[0], 1, 1, -1)
        x_l = x_g.clone()
        x_l[:, :, :, -1] = 0
        unfold = nn.Unfold(
            kernel_size=(1, self.hp.kernel_size),
            dilation=1,
            padding=0,
            stride=(1, self.hp.stride),
        )
        unfold_x = unfold(x_l)
        unfold_x = unfold_x.transpose(1, 2)
        f_local = torch.fft.rfft(unfold_x, dim=-1)
        f_local = torch.cat((f_local.real, f_local.imag), dim=-1)
        f_local = self.emb_local(f_local)
        for enc_layer in self.atten:
            f_local, enc_slf_attn = enc_layer(f_local)
        f_local = self.out_linear(f_local)
        f_local = f_local[:, -1, :]  # [batch_size, condition_emb_dim]

        output = torch.cat((f_global, f_local), dim=-1)  # [batch_size, condition_emb_dim + condition_emb_dim]
        output = F.layer_norm(output, [output.size(-1)])
        return output

    # (Other methods like MCMC2, loss_func, etc., remain unchanged)
    def MCMC2(self, x, y):
        condition = self.get_conditon(x)
        condition_flat = condition.view(x.size(0), -1)
        origin_x = x.clone()
        for _ in range(10):
            encoder_input = torch.cat([x.view(x.size(0), -1), condition_flat], dim=1)
            mu, var = self.encode(encoder_input)
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(z, condition_flat)
            recon = -0.5 * (torch.log(var_x + 1e-8) + (origin_x - mu_x) ** 2 / (var_x + 1e-8))
            temp = torch.quantile(recon, self.hp.mcmc_rate, dim=-1).unsqueeze(-1)
            l = (temp < recon).int()
            x = mu_x * (1 - l) + origin_x * l
        prob_all = 0
        for _ in range(128):
            mu, var = self.encode(torch.cat([x.view(x.size(0), -1), condition_flat], dim=1))
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(z, condition_flat)
            prob_all += -0.5 * (torch.log(var_x + 1e-8) + (origin_x - mu_x) ** 2 / var_x)
        return x, prob_all / 128

    def loss_func(self, mu_x, var_x, input, mu, var, y, z, mode="nottrain"):
        if mode == "train":
            self.num_iter += 1
            self.num_iter = self.num_iter % 100
        kld_weight = 0.005
        if self.loss_type != "M":
            mu_x = mu_x.squeeze(1)
            var_x = var_x.squeeze(1)
            input = input.squeeze(1)
        if self.loss_type == "M":
            var_x = var_x + 1e-8
            # 调试输出：var_x 形状及数值范围
            #print(
                #f"[DEBUG] var_x shape: {var_x.shape}, mean: {var_x.mean().item():.4f}, min: {var_x.min().item():.4f}, max: {var_x.max().item():.4f}")
            alpha = y.float().unsqueeze(1)  # 显式转换为浮点型
            # 调试输出：alpha 关键信息（是否与mask逻辑一致）
            #print(
                #f"[DEBUG] alpha shape: {alpha.shape}, mean: {alpha.mean().item():.4f}, True比例: {(alpha == 1).float().mean().item():.4f}, False比例: {(alpha == 0).float().mean().item():.4f}")
            # 重构项计算
            recon_term_numerator = (input - mu_x).pow(2)
            recon_term = alpha * (torch.log(var_x) + recon_term_numerator / var_x)
            # 调试输出：重构项分布
            #print(
                #f"[DEBUG] recon_term shape: {recon_term.shape}, mean: {recon_term.mean().item():.4f}, min: {recon_term.min().item():.4f}, max: {recon_term.max().item():.4f}")
            recon_loss = 0.5 * torch.mean(recon_term)
            #print(f"[DEBUG] recon_loss: {recon_loss.item():.6f}")  # 打印标量损失值
            # KL散度项计算
            kld_element = 1 + var - mu.pow(2) - var.exp()
            #print(
                #f"[DEBUG] kld_element shape: {kld_element.shape}, mean: {kld_element.mean().item():.4f}, min: {kld_element.min().item():.4f}, max: {kld_element.max().item():.4f}")
            kld_loss_per_sample = -0.5 * torch.sum(kld_element, dim=1)
            kld_loss = torch.mean(kld_loss_per_sample)
            #print(f"[DEBUG] kld_loss_per_sample shape: {kld_loss_per_sample.shape}, mean: {kld_loss.item():.6f}")
            # Beta系数（mask均值，反映有效数据比例）
            beta = torch.mean(y.float())
            #print(f"[DEBUG] beta (mask mean): {beta.item():.4f} （理论值：有效数据比例，全有效时应为1.0）")
            loss = recon_loss + beta * kld_loss
            #loss = loss.clamp(max=1e6)
            #print(f"[DEBUG] freq_loss: {loss.item():.6f}")
        else:
            recon_loss = torch.mean(
                0.5 * torch.mean(y * (torch.log(var_x) + (input - mu_x) ** 2 / var_x), dim=1),
                dim=0,
            )
            m = (torch.sum(y, dim=1, keepdim=True) / self.hp.window).repeat(
                1, self.hp.latent_dim
            )
            kld_loss = torch.mean(
                0.5 * torch.mean(m * (z ** 2) - torch.log(var) - (z - mu) ** 2 / var, dim=1),
                dim=0,
            )
            if self.loss_type == "B":
                self.C_max = self.C_max.to(input.device)
                C = torch.clamp(
                    self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0]
                )
                loss = recon_loss + self.gamma * kld_weight * (kld_loss - C).abs()
            elif self.loss_type == "C":
                loss = recon_loss + kld_loss
            elif self.loss_type == "D":
                loss = recon_loss + self.num_iter / 100 * kld_loss
            else:
                raise ValueError("Undefined loss type.")
            loss = loss.mean()
        return loss

    def _calc_conv_output_size(self):
        conv_out = (self.hp.window - self.hp.kernel_size) // self.hp.stride + 1
        assert conv_out > 0, f"Invalid conv params: kernel={self.hp.kernel_size}, stride={self.hp.stride}"
        self.conv_output_size = conv_out

    def encode_condition(self, time_feats):
        return self.condition_encoder(time_feats)

    def get_global_corr(self, x):
        if x.dim() == 3:  # [batch_size, num_channels, window]
            fft_x = torch.fft.rfft(x, dim=-1)  # [batch_size, num_channels, n_freq]
            corr = torch.abs(fft_x).mean(dim=[1, 2])  # 平均所有通道和频率
        else:
            fft_x = torch.fft.rfft(x, dim=-1)
            corr = torch.abs(fft_x).mean(dim=1)
        return corr


class TimeCVAE(nn.Module):
    def __init__(
            self,
            hp,
            num_stats_features,
            gamma: float = 1000.0,
            max_capacity: int = 25,
            Capacity_max_iter: int = 1e5,
            loss_type: str = "M",
            temp_mask_ratio: float = 0.25
    ):
        super(TimeCVAE, self).__init__()
        self.hp = hp
        self.loss_type = loss_type
        self.num_time_features = num_stats_features
        self.num_channels = hp.num_channels
        self.gamma = gamma
        self.dropout = nn.Dropout(self.hp.dropout_rate)
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.temp_mask_ratio = temp_mask_ratio
        self.input_features = None
        self.num_iter = 0
        self.temp_conv = None
        self.lstm = None

        # CNN for encoding
        self.encoder_conv = nn.Conv1d(
            in_channels=self.num_channels,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2
        )
        conv_out_size = self._calc_conv_output_size()
        self.encoder_fc = nn.Sequential(
            nn.Linear(64 * conv_out_size, 100),
            nn.Tanh()
        )
        # Adjusted encoder to accept 100 + 3 * condition_emb_dim
        condition_input_dim = 3 * hp.condition_emb_dim  # Reflects f_global, f_local, stats_proj
        self.encoder = nn.Sequential(
            nn.Linear(100 + condition_input_dim, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh()
        )
        self.fc_mu = nn.Linear(100, hp.latent_dim)
        self.fc_var = nn.Sequential(nn.Linear(100, hp.latent_dim), nn.Softplus())

        # Decoder adjusted to accept hp.latent_dim + 3 * condition_emb_dim
        condition_input_dim = 3 * hp.condition_emb_dim  # 192
        self.decoder = nn.Sequential(
            nn.Linear(hp.latent_dim + condition_input_dim, 256),  # 8 + 192 = 200
            nn.ReLU(),
            nn.Linear(256, hp.window)
        )
        self.fc_var_x = nn.Sequential(nn.Linear(hp.window, hp.window), nn.Softplus())

        # Conditioning components
        self.condition_encoder = nn.Sequential(
            nn.Linear(num_stats_features, hp.condition_emb_dim),
            nn.LayerNorm(hp.condition_emb_dim),
            nn.ReLU()
        )
        self.atten = nn.ModuleList([
            EncoderLayer_selfattn(hp.d_model, hp.d_inner, hp.n_head, dropout=0.1)
        ])
        self.out_linear = nn.Linear(hp.d_model, hp.condition_emb_dim)
        self.time_attn = EncoderLayer_selfattn(
            d_model=128, d_inner=hp.d_inner, n_head=hp.n_head, dropout=0.1
        )
        self.attn_norm = nn.LayerNorm(128)
        self.m_temp = nn.Parameter(torch.randn(self.hp.d_model, 1) * 0.01)
        self.dropout = nn.Dropout(hp.dropout_rate)

    def _calc_conv_output_size(self):
        # Output size remains same as input with kernel=5, stride=1, padding=2
        return self.hp.window

    def _dynamic_init(self, input):
        if input.dim() == 2:
            input = input.unsqueeze(1)
        B, C, L = input.shape
        if self.input_features != C or self.temp_conv is None:
            self.input_features = C
            self.temp_conv = nn.Conv1d(
                C, self.hp.d_model, kernel_size=self.hp.kernel_size, stride=self.hp.stride
            ).to(input.device)
            self.lstm = nn.LSTM(
                input_size=C, hidden_size=64, num_layers=1, bidirectional=True, batch_first=True
            ).to(input.device)

    def get_conditon(self, x):
        if self.temp_conv is None:
            self._dynamic_init(x)
        B, C, L = x.shape
        if C != self.input_features:
            x = x.mean(dim=1, keepdim=True)
        x_mean = x.mean(dim=2)  # [batch_size, num_channels]
        x_var = x.var(dim=2)
        stats_features = torch.cat([x_mean, x_var], dim=1)
        stats_proj = nn.Linear(2 * self.num_channels, self.hp.condition_emb_dim).to(x.device)(stats_features)

        # Convolutional features with reduced masking impact
        conv_out = self.temp_conv(x)
        anomaly_score = torch.var(conv_out, dim=1)
        k = max(int(self.temp_mask_ratio * 0.5 * conv_out.size(-1)), 1)  # Halve masking ratio
        _, mask_indices = torch.topk(anomaly_score, k, dim=-1)
        masked_conv_out = conv_out.clone()
        for b in range(B):
            masked_conv_out[b, :, mask_indices[b]] = self.m_temp.to(masked_conv_out.dtype)
        # Average masked and unmasked features for stability
        f_local = (conv_out + masked_conv_out) / 2.0
        f_local = f_local.transpose(1, 2)
        for enc_layer in self.atten:
            f_local, _ = enc_layer(f_local)
        f_local = self.out_linear(f_local[:, -1, :])

        f_global = torch.mean(x, dim=-1, keepdim=True).expand(-1, -1, self.hp.condition_emb_dim)
        output = torch.cat([f_global.squeeze(1), f_local, stats_proj], dim=-1)

        # Normalize conditioning output
        output = F.layer_norm(output, [output.size(-1)])  # Already correct as 2D, but more general
        return output

    def encode(self, input, condition):
        # Reshape for CNN: [batch, 1, window]
        input_conv = input.view(input.size(0), 1, self.hp.window)
        conv_out = self.encoder_conv(input_conv)  # [batch, 64, window]
        conv_flat = conv_out.view(input.size(0), -1)  # [batch, 64 * window]
        hidden = self.encoder_fc(conv_flat)  # [batch, 100]
        condition_flat = condition.view(input.size(0), -1)
        encoder_input = torch.cat([hidden, condition_flat], dim=1)  # [batch, 100 + 2*condition_emb_dim]
        hidden = self.encoder(encoder_input)  # [batch, 100]
        mu = self.fc_mu(hidden)
        var = self.fc_var(hidden)
        return mu, var

    def decode(self, z, condition):
        decoder_input = torch.cat([z, condition], dim=1)
        mu_x = self.decoder(decoder_input)
        var_x = self.fc_var_x(mu_x)
        return mu_x, var_x

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-7)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, mode, y):
        if mode in ["train", "valid"]:
            condition = self.get_conditon(input)
            condition_flat = condition.view(input.size(0), -1)
            mu, var = self.encode(input, condition)
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(z, condition_flat)
            rec_x = self.reparameterize(mu_x, var_x)
            loss = self.loss_func(mu_x, var_x, input, mu, var, y, z)
            return [mu_x, var_x, rec_x, mu, var, condition_flat, loss]
        else:
            return self.MCMC2(input, y)

    def MCMC2(self, x, y):
        if x.dim() == 3:
            x = x.squeeze(1)
        condition = self.get_conditon(x.unsqueeze(1))
        condition_flat = condition.view(x.size(0), -1)
        origin_x = x.clone()
        for _ in range(10):
            mu, var = self.encode(origin_x.unsqueeze(1), condition)
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(z, condition_flat)
            recon = -0.5 * (torch.log(var_x + 1e-8) + (origin_x - mu_x) ** 2 / (var_x + 1e-8))
            temp = torch.quantile(recon, self.hp.mcmc_rate, dim=-1).unsqueeze(-1)
            l = (temp < recon).int()
            origin_x = mu_x * (1 - l) + origin_x * l
        prob_all = 0
        for _ in range(128):
            mu, var = self.encode(origin_x.unsqueeze(1), condition)
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(z, condition_flat)
            prob_all += -0.5 * (torch.log(var_x + 1e-8) + (origin_x - mu_x) ** 2 / var_x)
        return origin_x, prob_all / 128

    def loss_func(self, mu_x, var_x, input, mu, var, y, z, mode="nottrain"):
        if mode == "train":
            self.num_iter += 1
            self.num_iter = self.num_iter % 100
        if self.loss_type != "M":
            mu_x = mu_x.squeeze(1)
            var_x = var_x.squeeze(1)
            input = input.squeeze(1)
        if self.loss_type == "M":
            var_x = var_x + 1e-8
            alpha = y.float().unsqueeze(1)
            # Smooth beta with exponential moving average (EMA)
            if not hasattr(self, 'beta_ema'):
                self.beta_ema = torch.mean(y.float()).item()
            self.beta_ema = 0.9 * self.beta_ema + 0.1 * torch.mean(y.float()).item()
            beta = torch.tensor(self.beta_ema, device=input.device)

            # Gradual KL weight ramp-up over 2000 iterations (approx. 20 epochs)
            kl_weight = min(1.0, self.num_iter / 2000.0) * 0.005

            # Reconstruction term with capping
            recon_term = alpha * (torch.log(var_x) + torch.clamp((input - mu_x).pow(2) / var_x, max=1e4))
            recon_loss = 0.5 * torch.mean(recon_term)

            # KL divergence term
            kld_element = 1 + var - mu.pow(2) - var.exp()
            kld_loss = -0.5 * torch.sum(kld_element, dim=1)
            kld_loss = torch.mean(kld_loss) * kl_weight

            # Stabilized custom loss
            loss = 2.0 * recon_loss + beta * kld_loss
            loss = torch.clamp(loss, max=1e6)  # Prevent explosion
        else:
            recon_loss = torch.mean(
                0.5 * torch.mean(y * (torch.log(var_x) + (input - mu_x) ** 2 / var_x), dim=1),
                dim=0,
            )
            m = (torch.sum(y, dim=1, keepdim=True) / self.hp.window).repeat(1, self.hp.latent_dim)
            kld_loss = torch.mean(
                0.5 * torch.mean(m * (z ** 2) - torch.log(var) - (z - mu) ** 2 / var, dim=1),
                dim=0,
            )
            if self.loss_type == "B":
                self.C_max = self.C_max.to(input.device)
                C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
                loss = recon_loss + self.gamma * 0.005 * (kld_loss - C).abs()
            elif self.loss_type == "C":
                loss = recon_loss + kld_loss
            elif self.loss_type == "D":
                loss = recon_loss + self.num_iter / 100 * kld_loss
            else:
                raise ValueError("Undefined loss type.")
            loss = loss.mean()
        return loss

    def get_global_corr(self, x):
        if x.dim() == 3:  # [batch_size, num_channels, window]
            autocorr = torch.zeros(x.size(0), x.size(2) - 1, device=x.device)
            for lag in range(1, x.size(2)):
                autocorr[:, lag - 1] = torch.sum(x[:, :, :-lag] * x[:, :, lag:], dim=2).mean(dim=1)
            return autocorr.mean(dim=1)
        else:
            autocorr = torch.zeros(x.size(0), x.size(1) - 1, device=x.device)
            for lag in range(1, x.size(1)):
                autocorr[:, lag - 1] = torch.sum(x[:, :-lag] * x[:, lag:], dim=1)
            return autocorr.mean(dim=1)