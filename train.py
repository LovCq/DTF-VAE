# train.py
from ast import arg
import os
import logging
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from model import MyVAE
from pytorch_lightning.loggers import TensorBoardLogger
import argparse

SEED = 8
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

logger = TensorBoardLogger(name="logs", save_dir="./")


def main(hparams):
    print("Loading model...")
    model = MyVAE(hparams)
    print("Model built")
    early_stop = EarlyStopping(
        monitor="val_loss_valid_epoch", patience=5, verbose=True, mode="min"
    )
    checkpoint = ModelCheckpoint(
        dirpath="./ckpt/",
        filename="{}".format(hparams.data_name),
        monitor="val_loss_valid_epoch",
        mode="min",
    )
    trainer = Trainer(
        max_epochs=hparams.max_epoch,
        callbacks=[early_stop, checkpoint],
        logger=logger,
        accelerator="gpu",
        devices=[hparams.gpu],
        check_val_every_n_epoch=1,
        gradient_clip_algorithm="value",
    )

    print("Fit start")
    trainer.fit(model)
    print("Testing start")
    trainer.test(model)

    # Print TensorBoard log instructions
    print("View tensorboard logs by running\ntensorboard --logdir %s" % os.getcwd())
    print("and going to http://localhost:6006 on your browser")


if __name__ == "__main__":
    parser = MyVAE.add_model_specific_args()
    hyperparams = parser.parse_args()
    print(f"RUNNING with hyperparameters: {hyperparams}")
    main(hyperparams)