import os
import torch

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH'] = torch.__version__
