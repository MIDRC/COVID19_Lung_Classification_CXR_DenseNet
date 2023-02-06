import os
from argparse import ArgumentParser
from tqdm import tqdm

import pydicom

import pandas as pd
import numpy as np

import cv2
from scipy import ndimage
from skimage.exposure import equalize_hist
from skimage.transform import resize
from PIL import Image

import torch
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from masks.mask import generate

def get_minimal_transform(img, threshold=125, mode='largest'):
    """Transformation to rotate and crop DICOM images to the smallest bounding box.

    Keyword arguments:
    img -- input image (h x w)
    threshold -- threshold for the binary mask
    mode -- largest or all contours
    """

    assert(mode in ['largest', 'all'])

    # convert to black/white
    _, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # convert to uint8 to work with the contour fn
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if mode == 'largest':
      areas = [cv2.contourArea(contour) for contour in contours]
      order = np.flip(np.argsort(areas))
      contour = contours[order[0]]

    elif mode == 'all':
      stacked_contours = np.vstack(contours)
      contour = cv2.convexHull(stacked_contours)

    # find the smallest rotated rect for the contour
    rect = cv2.minAreaRect(contour)
    center, (w, h), angle = rect

    # do not rotate the image
    width = np.ceil(w).astype(np.int32)
    height = np.ceil(h).astype(np.int32)
    if angle <= -45:
        angle += 90
    elif angle >= 45:
        angle -=90
        height = np.ceil(w).astype(int)
        width = np.ceil(h).astype(int)


    # rotate
    theta = angle * np.pi / 180 # convert to rad
    v_x = (np.cos(theta), np.sin(theta))
    v_y = (-np.sin(theta), np.cos(theta))

    # translate
    s_x = center[0] - v_x[0] * ((width-1) / 2) - v_y[0] * ((height-1) / 2)
    s_y = center[1] - v_x[1] * ((width-1) / 2) - v_y[1] * ((height-1) / 2)

    mapping = np.array([[v_x[0],v_y[0], s_x],
                        [v_x[1],v_y[1], s_y]])

    return mapping, width, height

def minimize(img, mapping, width, height):
    """Rotate and crop DICOM images to the smallest rotated bounding box.

    Keyword arguments:
    img -- input image
    mapping -- transformation to apply
    width -- width of the output image
    height -- height of the output image
    """
    return cv2.warpAffine(img, mapping, (width, height),
      flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def load_img(path):
  """Generic image loader.

    Keyword arguments:
    path -- path to an JPG or DCM image
  """

  ext = path[-3:].lower()
  assert(ext in ['dcm', 'jpg', 'dic'])

  if path[-3:].lower() in ['dcm', 'dic']:
    ds = pydicom.dcmread(path)
    img = ds.pixel_array

    # squeeze between 0 and 1 and invert if needed
    img = (img - img.min()) / (img.max() - img.min())
    if ds.PhotometricInterpretation == 'MONOCHROME1':
      img = 1 - img

  elif path[-3:].lower() == 'jpg':
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = (img - img.min()) / (img.max() - img.min())

  return img

def blend(img, mask, dilation=5):
  """Apply a dilated mask to the image.

    Keyword arguments:
    img -- a torch tensor (h x w)
    mask -- a binary mask (h x w)
    dilation -- number of pixels to dilate the mask with
  """

  # postprocess mask
  mask = torch.tensor(mask, dtype=torch.uint8).unsqueeze(0)
  mask = TF.resize(mask, size=img.shape, interpolation=Image.NEAREST)

  mask = mask.squeeze().numpy()
  mask = ndimage.binary_dilation(mask, iterations=dilation)

  # mask
  img = np.where(mask > 0, img, mask)

  return img


def preprocess(path, output_size=(320, 320)):
  """Preprocess pipeline for different sources of samples.

    Keyword arguments:
    path -- path to an JPG or DCM image
    output_size -- dimensions of the preprocessed image
  """
  img = load_img(path)

  # remove useless "padding"
  mapping, width, height = get_minimal_transform(img, threshold=0.5, mode='largest')
  img = minimize(img, mapping, width, height)

  # histogram equalization
  img = equalize_hist(img)

  # generate the lung mask and blend the img with the mask
  mask = generate(img)
  img = blend(img, mask)

  # minimize the result
  mapping, width, height = get_minimal_transform(img * 255, threshold=0, mode='all')
  img = minimize(img, mapping, width, height)

  # to tensor and resize
  img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
  img = TF.resize(img, size=output_size, interpolation=Image.BILINEAR)

  return img


def cli_main():
  # seed
  pl.seed_everything(365)

  # parse cmd-line arguments
  parser = ArgumentParser()
  parser.add_argument('--ricord', action='store_true', default=False)
  parser.add_argument('--chexpert', action='store_true', default=False)
  parser.add_argument('--iitac', action='store_true', default=False)
  parser.add_argument('--negative_sample_size', type=int, default=5000)
  parser.add_argument('--data_dir', type=str, default='.', help='Path to input files')
  parser.add_argument('--output_dir', type=str, default='./output')
  parser.add_argument('--device', type=str, default='cuda:1')
  args = parser.parse_args()

  assert(args.chexpert or args.ricord or args.iitac)

  # parse chexpert
  if args.chexpert:
    df = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
    df = df[df['Frontal/Lateral'] == 'Frontal'].sample(args.negative_sample_size).reset_index()
    df['relative_path'] = df['Path'].str.slice(20)

    for idx, row in tqdm(df.iterrows(), total=args.negative_sample_size):
      path = os.path.join(args.data_dir, row['relative_path'])
      img = preprocess(path)

      # write to disk
      output_path = os.path.join(args.output_dir, path[1:-4].replace('/', '-') + '.pt')
      torch.save(img, output_path)

  # parse ricord
  if args.ricord:
    df = pd.read_csv(os.path.join(args.data_dir, 'ricord.csv'), sep=';')

    for idx, row in tqdm(df.iterrows(), total=len(df)):
      path = os.path.join(args.data_dir, row['input_path'][25:])
      img = preprocess(path)

      # write to disk
      output_path = os.path.join(args.output_dir, row['SOPInstanceUID'] + '.pt')
      torch.save(img, output_path)

  # parse iitac
  if args.iitac:
    filenames = os.listdir(args.data_dir)
    for idx, fn in tqdm(enumerate(filenames), total=len(filenames)):
      path = os.path.join(args.data_dir, fn)
      img = preprocess(path)

      # write to disk
      output_path = os.path.join(args.output_dir, fn[:-4] + '.pt')
      torch.save(img, output_path)

if __name__ == '__main__':
    cli_main()