# For relative import
import os
import sys
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ_DIR)
print(PROJ_DIR)
import argparse

from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor

from models.ASTGCN import ASTGCN_Submodule
from models.JMGCN import JMGCN_submodule
from models.MSTGCN_lstm import MSTGCN_submodule_lstm
from models.STGCN import STGCN_submodule
from models.fusiongraph import FusionGraphModel
from datasets.air import *
from util import *


parser = argparse.ArgumentParser()
args = parser.parse_args()

gpu_num = 0                                                # set the GPU number of your server.
os.environ['WANDB_MODE'] = 'offline'                        # select one from ['online','offline']

hyperparameter_defaults = dict(
    server=dict(
        gpu_id=0,
    ),
    graph=dict(
        use=['poi', 'func', 'dist'],
        fix_weight=False,              # Fixed fusion.
        tempp_diag_zero=False,
        matrix_weight=True,            # Adaptive fusion for the three matrices.
    ),
    model=dict(
        # TODO: check batch_size
            use='JMGCN'
    ),
    data=dict(
        in_dim=7,
        out_dim=1,
        hist_len=12,
        pred_len=24,
        type='aqi',
    ),
    JMGCN=dict(
        use_fusion_graph=True,
    ),
    train=dict(
        seed=3407,  # 1997, 42,
        epoch=200,
        batch_size=256,
        lr=1e-4,
        weight_decay=1e-4,
        M=28,
        d=7,
        bn_decay=0.1,
    )
)

wandb_proj = 'aqi'
wandb.init(config=hyperparameter_defaults, project=wandb_proj)
wandb_logger = WandbLogger()
config = wandb.config

pl.utilities.seed.seed_everything(config['train']['seed'])

gpu_id = config['server']['gpu_id']
device = 'cuda:%d' % gpu_id

if config['data']['type'] == 'aqi':
    root_dir = 'data'
    pm25_data_dir = os.path.join(root_dir, 'temporal_data')
    pm25_graph_dir = os.path.join(root_dir, 'matrix')
else:
    raise NotImplementedError

if config['data']['type'] == 'aqi':
    graph = AirGraph(pm25_graph_dir, config['graph'], gpu_id)
    train_set = Air(pm25_data_dir, 'train')
    val_set = Air(pm25_data_dir, 'val')
    test_set = Air(pm25_data_dir, 'test')
else:
    raise NotImplementedError

scaler = train_set.scaler


class LightningData(LightningDataModule):
    def __init__(self, train_set, val_set, test_set):
        super().__init__()
        self.batch_size = config['train']['batch_size']
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=12,
                                   pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=8,
                                 pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=8,
                                  pin_memory=True, drop_last=True)


class LightningModel(LightningModule):
    def __init__(self, scaler, fusiongraph):
        super().__init__()

        self.scaler = scaler
        self.fusiongraph = fusiongraph

        self.metric_lightning = LightningMetric()

        if config['model']['use'] == 'ASTGCN':  # baselines
            self.model = ASTGCN_Submodule(gpu_id, fusiongraph, config['data']['in_dim'],
                                          config['data']['hist_len'], config['data']['pred_len'])
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)
        elif config['model']['use'] == 'MSTGCN_lstm':
            self.model = MSTGCN_submodule_lstm(gpu_id, fusiongraph, config['data']['in_dim'],
                                               config['data']['hist_len'], config['data']['pred_len'])
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)
        elif config['model']['use'] == 'JMGCN':  # our model
            self.model = JMGCN_submodule(gpu_id, fusiongraph, config['data']['in_dim'],
                                         config['data']['hist_len'], config['data']['pred_len'])
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)
        elif config['model']['use'] == 'STGCN':
            self.model = STGCN_submodule(gpu_id, fusiongraph, config['data']['in_dim'], config['data']['hist_len'],
                                         config['data']['pred_len'])
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)

        else:
            raise NotImplementedError

        self.log_dict(config)

    def forward(self, x):
        return self.model(x)

    def _run_model(self, batch):
        x, y = batch
        y_hat = self(x)
        y_hat = self.scaler.inverse_transform(y_hat)
        # loss = masked_mae(y_hat, y, 0.0)
        loss = masked_huber(y_hat, y, 0.0)
        return y_hat, y, loss

    def training_step(self, batch, batch_idx):
        y_hat, y, loss = self._run_model(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y, loss = self._run_model(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        y_hat, y, loss = self._run_model(batch)
        self.metric_lightning(y_hat.cpu(), y.cpu())
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_epoch_end(self, outputs):
        test_metric_dict = self.metric_lightning.compute()
        self.log_dict(test_metric_dict)

    # revise
    def configure_optimizers(self):

        # revised 当10个epoch上性能不提升时，自动缩小学习率
        optimizer = torch.optim.Adam(self.parameters(), lr=config['train']['lr'],
                                     weight_decay=config['train']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.5, patience=75, min_lr=1e-7)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


def main():
    fusiongraph = FusionGraphModel(graph, gpu_id, config['graph'], config['data'], config['train']['M'],
                                   config['train']['d'], config['train']['bn_decay'])

    lightning_data = LightningData(train_set, val_set, test_set)

    lightning_model = LightningModel(scaler, fusiongraph)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(
        logger=wandb_logger,
        gpus=[gpu_id],
        max_epochs=config['train']['epoch'],
        callbacks=[lr_monitor]  # revise
    )

    trainer.fit(lightning_model, lightning_data)
    trainer.test(lightning_model, datamodule=lightning_data)
    print('Graph USE', config['graph']['use'])
    print('Data', config['data'])


if __name__ == '__main__':
    main()
