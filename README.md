# DTF-VAE: A Self-Adaptive Time-Frequency Fusion VAE for Robust Time Series Anomaly Detection

[![Paper](https://img.shields.io/badge/Paper-arxiv.XXXX.XXXXX-red)](https://arxiv.org/abs/xxxx.xxxxx)
[![GitHub](https://img.shields.io/github/license/LovCq/DTF-VAE)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

Official PyTorch implementation of **DTF-VAE** as described in the paper:  
*DTF-VAE: A Self-Adaptive Time-Frequency Fusion VAE for Robust Time Series Anomaly Detection*  
Taixiang Wang, Yong Zhou, Jiawei Xu, Yangsiyu Zhang  
Dalian University of Technology, China  

---

## 📖 Abstract

Time series anomaly detection is crucial for applications such as industrial monitoring and financial risk management. A practical detector needs to capture both transient deviations and periodic irregularities, which are often more salient in the time and frequency domains, respectively. However, many existing time-frequency approaches rely on static fusion, which cannot adjust the relative importance of the two domains as anomaly types vary over time. In addition, reconstruction-based generative models often yield weakly separable latent representations, making subtle anomalies difficult to distinguish from normal fluctuations, especially with noisy or partially missing data.

To address these issues, we propose **DTF-VAE** (Dynamic Time-Frequency Variational Autoencoder), an unsupervised time-frequency variational framework for anomaly detection that does not require ground-truth anomaly labels. DTF-VAE models temporal and spectral features through two dedicated variational autoencoders and integrates them via a dynamic fusion module that combines cross-attention-based alignment with adaptive gating to balance the contributions of each modality for each input instance. We further optimize a composite objective that couples reconstruction and latent regularization with correlation- and contrastive-based terms constructed from self-supervised augmentations during training, to improve anomaly sensitivity and latent separability.

Extensive experiments on four public benchmarks show that DTF-VAE consistently outperforms strong baselines. Additional evaluations under controlled noise and missing-data settings confirm its robustness.

---

## 🚀 Key Contributions

1. **Dynamic Time-Frequency Fusion** – A cross-modal alignment and adaptive gating mechanism that adjusts modality contributions on a per-window basis, overcoming the limitations of static fusion.
2. **Composite Objective** – Combines reconstruction, KL regularization, correlation-based dependency preservation, and triplet contrastive learning to enhance latent separability and anomaly sensitivity.
3. **State-of-the-Art Performance** – Achieves best F1 scores on Yahoo, KPI, WSD, and NAB benchmarks under both point-adjust and delay-tolerant evaluation protocols.
4. **Robustness** – Extensive ablation and robustness studies demonstrate stable performance under Gaussian noise and random missing data.


## 🛠️ Installation

```bash
git clone https://github.com/LovCq/DTF-VAE.git
cd DTF-VAE
pip install -r requirements.txt
