import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import metrics
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from models import DenseNet, ResNet
from utils import rename_weights
from dataset import MIDRCDataModule

# --------------
# Loss
# --------------

class BinaryFocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=0):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-6

    # ones or zeros
    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.eps, 1.0 - self.eps)
        alpha = torch.where(target == 1, torch.full_like(target, self.alpha), torch.ones_like(target))
        pt = torch.where(target == 1, prob, 1 - prob)
        fl = -alpha * torch.pow(1 - pt, self.gamma) * torch.log(pt)
        return torch.mean(fl)

# --------------
# PL Module
# --------------

class MIDRCNet(pl.LightningModule):
    def __init__(self,
        backbone='densenet',
        freeze=False,
        clf_features=0,
        batch_size=16,
        learning_rate=1e-4,
        aux_lambd=0.,
        weights='None'):
        super(MIDRCNet, self).__init__()
        self.save_hyperparameters()

        # weights
        wts = None
        if weights and weights != 'None':
            wts = torch.load(
                os.path.join(os.getcwd(), weights),
                map_location=torch.device('cpu'))

        # classifier
        intermediate_features = None if clf_features == 0 else clf_features

        # densenet
        if backbone == 'densenet':
            # weights
            if wts != None:
                wts = rename_weights(
                    weights=wts['model_state'],
                    pattern='module.model.',
                    remove=['classifier.weight', 'classifier.bias'],
                    densenet=True)

            # setup the backbone
            arch = DenseNet(weights=wts, clf_features=intermediate_features)

            # freeze layers if necessary
            if freeze:
                for feature in arch.model.features.parameters():
                    feature.requires_grad = False

        # resnet
        if backbone == 'resnet':
            # setup the backbone
            arch = ResNet(weights=wts, clf_features=intermediate_features)

            # freeze layers if necessary
            if freeze:
                for feature in arch.model.parameters():
                    feature.requires_grad = False

        # set the backbone
        self.backbone = arch
        self.criterion = torch.nn.BCEWithLogitsLoss()
        # self.criterion = BinaryFocalLoss(alpha=3.5, gamma=2.5)

        # metrics
        self.train_acc1 = metrics.Accuracy()
        self.val_acc1 = metrics.Accuracy()
        self.test_acc1 = metrics.Accuracy()
        self.train_acc2 = metrics.Accuracy()
        self.val_acc2 = metrics.Accuracy()
        self.test_acc2 = metrics.Accuracy()
        self.train_acc3 = metrics.Accuracy()
        self.val_acc3 = metrics.Accuracy()
        self.test_acc3 = metrics.Accuracy()

    def forward(self, x):
        # inference/predictions
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        # compute the loss
        x, y1, y2, y3 = batch
        out1, out2, out3 = self.backbone(x)
        out1 = out1.squeeze()
        loss1 = self.criterion(out1, y1)
        loss2 = F.cross_entropy(out2, y2) * self.hparams.aux_lambd
        loss3 = F.cross_entropy(out3, y3) * self.hparams.aux_lambd
        loss = loss1 + loss2 + loss3

        # metrics
        self.log('train_acc1', self.train_acc1(out1, y1))
        self.log('train_acc2', self.train_acc2(out2, y2))
        self.log('train_acc3', self.train_acc3(out3, y3))
        self.log('train_loss1', loss1, on_step=False, on_epoch=True)
        self.log('train_loss2', loss2, on_step=False, on_epoch=True)
        self.log('train_loss3', loss3, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # compute the loss
        x, y1, y2, y3 = batch
        out1, out2, out3 = self.backbone(x)
        out1 = out1.squeeze()
        loss1 = self.criterion(out1, y1)
        loss2 = F.cross_entropy(out2, y2) * self.hparams.aux_lambd
        loss3 = F.cross_entropy(out3, y3) * self.hparams.aux_lambd
        loss = loss1 + loss2 + loss3

        # metrics
        self.log('val_acc1', self.val_acc1(out1, y1))
        self.log('val_acc2', self.val_acc2(out2, y2))
        self.log('val_acc3', self.val_acc3(out3, y3))
        self.log('val_loss1', loss1, on_step=False, on_epoch=True)
        self.log('val_loss2', loss2, on_step=True, on_epoch=True)
        self.log('val_loss3', loss3, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # compute the loss
        x, y1, y2, y3 = batch
        out1, out2, out3 = self.backbone(x)
        out1 = out1.squeeze()
        loss1 = self.criterion(out1, y1)
        loss2 = F.cross_entropy(out2, y2) * self.hparams.aux_lambd
        loss3 = F.cross_entropy(out3, y3) * self.hparams.aux_lambd
        loss = loss1 + loss2 + loss3

        # metrics
        self.log('test_acc1', self.test_acc1(out1, y1))
        self.log('test_acc2', self.test_acc2(out2, y2))
        self.log('test_acc3', self.test_acc3(out3, y3))
        self.log('test_loss1', loss1)
        self.log('test_loss2', loss2)
        self.log('test_loss3', loss3)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--backbone', default='densenet', type=str)
        parser.add_argument('--freeze', action='store_true', default=False)
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--aux_lambd', default=0., type=float)
        parser.add_argument('--clf_features', default=0, type=int)
        parser.add_argument('--weights', default='None', type=str)
        return parser

# --------------
# Training
# --------------

def cli_main():
    pl.seed_everything(365)

    # parser
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MIDRCNet.add_model_specific_args(parser)
    args = parser.parse_args()

    # data module
    dm = MIDRCDataModule(batch_size=args.batch_size, augment=True)

    # setup trainer
    model = MIDRCNet(
        backbone=args.backbone,
        weights=args.weights,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        aux_lambd=args.aux_lambd,
        clf_features=args.clf_features,
        freeze=args.freeze)

    checkpoint = ModelCheckpoint(
        dirpath='checkpoints',
        filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}',
        prefix='MIDRCNet',
        save_last=True,
        save_top_k=2,
        save_weights_only=True,
        monitor='val_loss')

    # logging
    # logger = WandbLogger(project='MIDRCNet')
    logger = TensorBoardLogger('tb_logs', name='MIDRCNet')

    # trainer
    trainer = pl.Trainer(
        # fast_dev_run=True,
        # overfit_batches=2,
        gpus=[1],
        max_epochs=100,
        callbacks=[checkpoint],
        logger=logger)

    trainer.fit(model, dm)

    # test
    result = trainer.test(datamodule=dm)
    print(result)

if __name__ == '__main__':
    cli_main()
