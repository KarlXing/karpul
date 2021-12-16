# File: save.py
# Author: Jinwei Xing (jinweixing1006@gmail.om)
# Last Modified: Wednesday, 15th December 2021
# This file is a part of karpul and distributed under MIT license.

import json

def save_dicts(path, dicts, indent=4):
    '''
    configs: a list of dicts to save
    '''
    with open(path, 'w') as f:
        if isinstance(dicts, list):
            for dict in dicts:
                json.dump(dict, f, indent=indent)
                f.write('\n')
        else:
            json.dump(dicts, f, indent=indent)

