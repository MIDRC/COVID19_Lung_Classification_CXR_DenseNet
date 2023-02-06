import re

def rename_weights(weights, pattern=None, replacement='', remove=[], densenet=False):
  """Rename the weights keys within a state dict.

    Keyword arguments:
    weights -- pretrained weights (e.g. CheXpert)
    pattern -- keys to find
    replacement -- replacement for the keys
    remove -- list of keys to remove from the weights
    densenet -
  """

  # replace a pattern with a replacement
  if pattern:
    for key in list(weights.keys()):
      weights[key.replace(pattern, replacement)] = weights.pop(key)

  # weights from densenets are renamed since '.' are not allowed in modulenames anymore
  if densenet:
    for key in list(weights.keys()):
      pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
      res = pattern.match(key)

      if res:
        new_key = res.group(1) + res.group(2)
        weights[new_key] = weights[key]
        del weights[key]

  # remove keys if necessary
  if len(remove) > 0:
    for key in remove:
      del weights[key]

  return weights