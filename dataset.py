import os
import random

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
import pytorch_lightning as pl

class MIDRCDataset(Dataset):
  """A dataset consisting of COVID and non-COVID chest x-rays based on CheXpert and RICORD"""

  def __init__(self, root='/data/sluijs/ricord', split='train', augment=False):
    """Create a dataset of COVID+/- chest x-rays and split into train, val and test.

      Keyword arguments:
      root -- directory including the data and metadata
      split -- either a train, val, or test split of the data
      augment -- apply random transformations to the chest x-ray
    """

    assert(split in ['train', 'val', 'test'])

    # setup
    self.augment = augment

    # negative cases
    neg_paths = [os.path.join(root, 'neg', fn) for fn in os.listdir(os.path.join(root, 'neg'))]
    neg_df = pd.DataFrame({
        'output_path': neg_paths,
        'label': 0.,
        'lesions': 'Negative',
        'severity': 'Negative' })

    # positive cases
    pos_df = pd.read_csv(os.path.join(root, 'ricord.csv'), sep=';').assign(label=1.)
    pos_df['output_path'] = pos_df['SOPInstanceUID'] \
      .apply(lambda x: os.path.join(root, 'pos', x + '.pt'))

    # remove incorrect cxrs (= not cxr)
    pos_df = pos_df[~pos_df['SOPInstanceUID'].isin([
      '1.2.826.0.1.3680043.10.474.2970653035463766670333181491641400321',
      '1.2.826.0.1.3680043.10.474.1131305637270336748148978173521989752',
      '1.2.826.0.1.3680043.10.474.1408706112518068608591514817243944042',
      '1.2.826.0.1.3680043.10.474.1644559290522038589787131735364516794',
      '1.2.826.0.1.3680043.10.474.2848922298608546673100466321882355212',
    ])]

    # splits
    neg_train, neg_val, neg_test = np.split(
      neg_df.sample(frac=1),
      [int(.7 * len(neg_df)), int(.86 * len(neg_df))],
    )

    pos_test = pos_df[pos_df['BitsStored'] == 15]
    pos_train, pos_val = np.split(
      pos_df[pos_df['BitsStored'] != 15].sample(frac=1),
      [int(.814 * (len(pos_df) - len(pos_test)))]
    )

    # combine train, val, test sets and reshuffle
    train = neg_train.append(pos_train).sample(frac=1).reset_index(drop=True)
    val = neg_val.append(pos_val).sample(frac=1).reset_index(drop=True)
    test = neg_test.append(pos_test).sample(frac=1).reset_index(drop=True)
    self.df = dict(train=train, val=val, test=test).get(split)

  def transform_random(self, img):
    """Randomly transform an input image.

      Keyword arguments:
      img -- tensor (c x h x w)
    """

    angle = [90., 180., 270., 360.][random.randint(0, 3)]
    blur = bool(np.random.binomial(1, 0.25))
    contrast = random.uniform(0.9, 1.5)
    brightness = random.uniform(0.5, 1.0)

    # rotation
    img = TF.affine(img, angle=angle, translate=[0, 0], scale=1., shear=[0, 0])

    # blur
    if blur:
      img = TF.gaussian_blur(img, kernel_size=9)

    # contrast/brightness
    img = TF.adjust_contrast(img, contrast_factor=contrast)
    img = TF.adjust_brightness(img, brightness_factor=brightness)

    return img

  def __len__(self):
    """Return the number of samples in the dataset."""

    return len(self.df)

  def __getitem__(self, idx):
    """Generate a tuple of a lung chest x-ray and its associated labels.

      Keyword arguments:
      idx -- index of the chest x-ray in the dataset split
    """

    # image
    path = self.df['output_path'].iloc[idx]
    img = torch.load(path)
    img = img.type(torch.float32).expand(3, img.shape[1], img.shape[2])

    if self.augment:
      img = self.transform_random(img)

    # covid/non-covid
    label = torch.tensor(self.df['label'].iloc[idx])

    # typicality
    typicality = pd.get_dummies(self.df['lesions']) \
        .reindex(['Negative', 'Atypical', 'Indeterminate', 'Typical'], axis=1) \
        .iloc[idx].to_numpy().astype(np.float32)
    typicality = torch.tensor(typicality).argmax(dim=0)

    # severity
    severity = pd.get_dummies(self.df['severity']) \
        .reindex(['Negative', 'Mild', 'Moderate', 'Severe'], axis=1) \
        .iloc[idx].to_numpy().astype(np.float32)
    severity = torch.tensor(severity).argmax(dim=0)

    return (img, label, typicality, severity)

class MIDRCDataModule(pl.LightningDataModule):
  """A reusable Pytorch Lightning Data Module"""

  def __init__(self, batch_size=32, augment=True):
    """Initialize the Data Module.

      Keyword arguments:
      batch_size -- number of samples within a mini-batch
      augment -- flag whether to apply augmentation to the training samples
    """

    super().__init__()
    self.batch_size = batch_size
    self.augment = augment

  def setup(self, *args):
    """Generate the training, validation and test splits."""

    self.train = MIDRCDataset(split='train', augment=self.augment)
    self.val = MIDRCDataset(split='val')
    self.test = MIDRCDataset(split='test')

  def train_dataloader(self):
    """Create a train dataloader."""

    return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

  def val_dataloader(self):
    """Create a val dataloader."""

    return DataLoader(self.val, batch_size=self.batch_size)

  def test_dataloader(self):
    """Create a test dataloader."""

    return DataLoader(self.test, batch_size=self.batch_size)


