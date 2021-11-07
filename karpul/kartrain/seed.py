# File: seed.py
# Author: Jinwei Xing (jinweixing1006@gmail.om)
# Last Modified: Sunday, 7th November 2021
# This file is a part of karpul and distributed under MIT license.

import random
import os


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        print('Seed Numpy Failure: numpy is not installed')

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        print('Seed Torch Failure: torch is not installed')