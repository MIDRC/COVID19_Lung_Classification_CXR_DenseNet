import os
import sys
from tqdm import tqdm
from argparse import ArgumentParser

import pandas as pd

import torch
import pytorch_lightning as pl

from train import MIDRCNet
from dataset import MIDRCDataModule

# --------------
# Predictions
# --------------

def cli_main():
  pl.seed_everything(365)

  # parser
  parser = ArgumentParser()
  parser.add_argument('--ckpt', type=str, default='')
  parser.add_argument('--output_path', type=str, default='./predictions/predictions.csv')
  args = parser.parse_args()

  # device
  device = torch.device('cuda:1')

  # model
  if not os.path.exists(os.path.join(os.getcwd(), args.ckpt)):
    sys.exit('Checkpoint [{}] not found.'.format(args.ckpt))

  model = MIDRCNet.load_from_checkpoint(
    checkpoint_path=os.path.join(os.getcwd(), args.ckpt),
    map_location=device).to(device)
  model.eval()

  # data
  dm = MIDRCDataModule(batch_size=1)
  dm.setup('test')

  rows = []
  for batch in tqdm(dm.test_dataloader(), total=len(dm.test)):
    x, y1, _, _ = batch

    # move to gpu
    x = x.to(device)
    y1 = y1.to(device)

    # predict
    out1, _, _ = model(x)
    y_hat1 = torch.sigmoid(out1).round().squeeze(dim=0)

    # save predictions
    for row in zip(y1.cpu().tolist(), y_hat1.cpu().tolist()):
      rows.append({
        'y1': row[0],
        'y_hat1': row[1],
      })

  # write to disk
  pd.DataFrame(rows).to_csv(args.output_path)

if __name__ == '__main__':
    cli_main()