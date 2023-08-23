"""The utils for llama model and generator"""

import torch.cuda as cuda

def get_local_device():
  if cuda.is_available():
    return "cuda"
  else:
    return "cpu"
