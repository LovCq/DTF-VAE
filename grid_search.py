import argparse
import itertools
import pandas as pd
from model import MyVAE
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import numpy as np
import os

# 定义超参数网格
param_grid = {
    'learning_rate': [0.0001, 0.0005, 0.001],
    'batch_size': [256, 512, 1024],
    'window': [48, 64, 128],
    'latent_dim': [4, 8, 16],
    'condition_emb_dim': [32, 64, 128],
    'n_head': [4, 8, 12],
    'kernel_size': [16, 24, 32],
    'stride': [8, 16, 24],
    'dropout_rate': [0.0, 0.05, 0.1],
    'missing_data_rate': [0.001, 0.005],
    'point_ano_rate': [0.01, 0.02],
    'seg_ano_rate': [0.02, 0.05],
    'contrast_weight': [0.1, 0.2],
    'corr_weight': [0.2, 0.5],
    'kld_weight': [0.001, 0.005]
}

# 生成所有超参数组合
keys, values = zip(*param_grid.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]


def run_experiment(base_hparams, exp_params):
    """
    运行单个实验，训练并测试模型，返回性能指标。

    Args:
        base_hparams (argparse.Namespace): 基础超参数
        exp_params (dict): 实验特定的超参数

    Returns:
        dict: 包含参数和性能指标的结果
    """
    # 合并基础参数和实验参数
    hparams = argparse.Namespace(**vars(base_hparams))
    for key, value in exp_params.items():
        setattr(hparams, key, value)

    # 设置随机种子
    torch.manual_seed(8)
    np.random.seed(8)
    torch.cuda.manual_seed_all(8)

    # 创建模型
    model = MyVAE(hparams)

    # 配置日志记录
    exp_name = '_'.join([f"{k}_{v}" for k, v in exp_params.items()])
    logger = TensorBoardLogger(
        save_dir="./logs",
        name="grid_search",
        version=exp_name
    )

    # 配置回调
    early_stop = EarlyStopping(
        monitor="val_loss_valid_epoch",
        patience=5,
        verbose=True,
        mode="min"
    )
    checkpoint = ModelCheckpoint(
        dirpath="./ckpt/",
        filename=f"{hparams.data_name}_{exp_name}",
        monitor="val_loss_valid_epoch",
        mode="min"
    )

    # 创建 Trainer
    trainer = Trainer(
        max_epochs=hparams.max_epoch,
        callbacks=[early_stop, checkpoint],
        logger=logger,
        accelerator="gpu",
        devices=[hparams.gpu],
        check_val_every_n_epoch=1,
        gradient_clip_algorithm="value",
        enable_progress_bar=True
    )

    # 训练和测试
    trainer.fit(model)
    trainer.test(model)

    # 获取测试指标
    metrics = model.test_metrics
    return {
        'params': exp_params,
        'auc': metrics.get('Metrics/AUC', 0.0),
        'best_f1': metrics.get('Metrics/Best_F1', 0.0),
        'delay_f1': metrics.get('Metrics/Delay_F1', 0.0)
    }


if __name__ == "__main__":
    # 解析基础参数
    parser = MyVAE.add_model_specific_args()
    base_hparams = parser.parse_args()

    # 设置固定的基础参数
    base_hparams.max_epoch = 30
    base_hparams.gpu = 0
    base_hparams.data_name = "your_data_name"  # 请根据实际数据集调整
    base_hparams.data_dir = "./data/Yahoo/"  # 请根据实际路径调整
    base_hparams.use_label = 1
    base_hparams.sliding_window_size = 1
    base_hparams.data_pre_mode = 0
    base_hparams.only_test = 0
    base_hparams.num_workers = 0
    # 可根据需要设置其他固定参数

    # 初始化结果列表
    results = []
    result_file = "grid_search_results.csv"

    # 清空或初始化结果文件（可选）
    if os.path.exists(result_file):
        os.remove(result_file)

    # 执行网格搜索
    for i, exp in enumerate(experiments):
        print(f"Running experiment {i + 1}/{len(experiments)} with params: {exp}")
        try:
            result = run_experiment(base_hparams, exp)
            results.append(result)
            # 保存中间结果
            result_df = pd.DataFrame([result])
            result_df.to_csv(
                result_file,
                mode='a',
                header=not os.path.exists(result_file),
                index=False
            )
            print(
                f"Completed: AUC={result['auc']:.4f}, Best_F1={result['best_f1']:.4f}, Delay_F1={result['delay_f1']:.4f}")
        except Exception as e:
            print(f"Experiment failed with error: {e}")
            continue

    # 分析结果
    if results:
        df_results = pd.DataFrame(results)
        best_exp = df_results.loc[df_results['best_f1'].idxmax()]
        print("\n=== Grid Search Completed ===")
        print(f"Best parameters: {best_exp['params']}")
        print(f"Best AUC: {best_exp['auc']:.4f}")
        print(f"Best F1: {best_exp['best_f1']:.4f}")
        print(f"Best Delay F1: {best_exp['delay_f1']:.4f}")
    else:
        print("No experiments completed successfully.")