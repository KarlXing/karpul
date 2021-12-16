# File: save.py
# Author: Jinwei Xing (jinweixing1006@gmail.om)
# Last Modified: Wednesday, 15th December 2021
# This file is a part of karpul and distributed under MIT license.

import json

def save_workspace(path, dicts, git_path=None, indent=4):
    '''
    configs: a list of dicts to save
    '''
    with open(path, 'w') as f:
        # save dicts
        if isinstance(dicts, list):
            for dict in dicts:
                json.dump(dict, f, indent=indent)
                f.write('\n')
        else:
            json.dump(dicts, f, indent=indent)

        # save git
        if git_path is not None:
            try:
                import git
                repo = git.Repo.init(git_path)
                commit = repo.head.commit.hexsha
                f.write('\ngit commit: %s \n' % (commit))
            except ImportError:
                print('Save Workspace Failure: gitpython is not installed')

