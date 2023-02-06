import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

# --------------
# Models: DenseNet-121
# --------------

class DenseNet(nn.Module):
  def __init__(self, in_channels=3, weights=None, clf_features=None, imagenet_pretraining=True):
    """Create a DenseNet-121.

      Keyword arguments:
      in_channels -- number of channels for the first convolutional layer
      weights -- load pretrained weights
      clf -- add a classifier on top of the backbone
      imagenet_pretraining -- load pretrained ImageNet weights
    """

    super().__init__()
    self.model = models.densenet121(pretrained=imagenet_pretraining)
    del self.model.classifier

    # replace number of input channels
    if in_channels != 3:
        self.model.features.conv0 = nn.Conv2d(
          in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # load weights
    if weights:
        self.model.load_state_dict(weights, strict=False)

    # create classifiers
    self.clf_label = create_classifier(1024, 1, clf_features) if clf_features else nn.Linear(1024, 1, bias=True)
    self.clf_typicality = create_classifier(1024, 4, clf_features) if clf_features else nn.Linear(1024, 4, bias=True)
    self.clf_severity = create_classifier(1024, 4, clf_features) if clf_features else nn.Linear(1024, 4, bias=True)

  def forward(self, x):
    features = self.model.features(x)
    out = F.relu(features, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    lbl = self.clf_label(out)
    typ = self.clf_typicality(out)
    sev = self.clf_severity(out)
    return (lbl, typ, sev)

# --------------
# Models: ResNet-50
# --------------

class ResNet(nn.Module):
  def __init__(self, in_channels=3, weights=None, clf_features=None, imagenet_pretraining=True):
    """Create a ResNet-121.

      Keyword arguments:
      in_channels -- number of channels for the first convolutional layer
      weights -- load pretrained weights
      clf -- add a classifier on top of the backbone
      imagenet_pretraining -- load pretrained ImageNet weights
    """

    super().__init__()
    features = nn.ModuleList(models.resnet50(pretrained=imagenet_pretraining).children())[:-1]
    self.model = nn.Sequential(*features)

    # replace layers
    if in_channels != 3:
        self.model[0] = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    if weights:
        self.model.load_state_dict(weights, strict=False)

    # create classifiers
    self.clf_label = create_classifier(2048, 1, clf_features) if clf_features else nn.Linear(2048, 1, bias=True)
    self.clf_typicality = create_classifier(2048, 4, clf_features) if clf_features else nn.Linear(2048, 4, bias=True)
    self.clf_severity = create_classifier(2048, 4, clf_features) if clf_features else nn.Linear(2048, 4, bias=True)

  def forward(self, x):
    out = self.model(x)
    out = torch.flatten(out, 1)
    lbl = self.clf_label(out)
    typ = self.clf_typicality(out)
    sev = self.clf_severity(out)
    return (lbl, typ, sev)

# --------------
# Models: Classifier
# --------------

def create_classifier(in_features=2048, num_classes=1, intermediate_features=None):
  """Create two-layer MLP classifier on top of the backbone.

    Keyword arguments:
    in_features -- number of features coming into the classifier
    num_classes -- number of out_classes
    intermediate_features -- number of features for the intermediate layer
  """

  if intermediate_features is None:
    intermediate_features = in_features // 2

  return nn.Sequential(
    nn.Linear(in_features, intermediate_features, bias=True),
    nn.ReLU(),
    nn.Linear(intermediate_features, num_classes, bias=True))

