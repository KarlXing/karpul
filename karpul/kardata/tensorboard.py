# File: tensorboard.py
# Author: Jinwei Xing (jinweixing1006@gmail.om)
# Last Modified: Wednesday, 3rd November 2021
# This file is a part of karpul and distributed under MIT license.

import glob
import numpy as np
import tensorboard as tb
from tensorboard.backend.event_processing import event_accumulator as ea

def smooth_scalars(scalars, weight):
    '''smooth scalars with smooth weight ranging from 0 to 1'''

    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  
        smoothed.append(smoothed_val)                        
        last = smoothed_val                                  

    return smoothed

def read_scalar_tag(event, scalar_tag, smooth_weight=0, interp=False, interp_x=None):
    '''read a scalar tag from an event'''

    # 1. sanity check
    assert(smooth_weight >= 0)
    if smooth_weight > 0:
        assert(interp is True)
        
    # 2. read tags from loaded event
    res = [(s.step, s.value) for s in event.Scalars(scalar_tag)]
    step, val = list(zip(*res))
    
    # 3. conduct interpolation and smooth if needed
    if interp and interp_x is not None:
        step, val = list(interp_x), np.interp(interp_x, step, val)
        if smooth_weight > 0:
            val = smooth_scalars(val, smooth_weight)
    
    # 4. return result
    return (step, val)
    
def read_event_tags(scalar_tags, event_file, verbal=True, smooth_weight=0, interp=False, interp_x=None):
    '''read multiple scalar tags from one event'''

    # 1. load event
    if verbal:
        print('start processing %s' % event_file)
    event = ea.EventAccumulator(event_file)
    event.Reload()
    
    # 2. read tags
    res = {}
    for tag in scalar_tags:
        res[tag] = read_scalar_tag(event, tag, smooth_weight=smooth_weight, interp=interp, interp_x=interp_x)
    
    # 3. return
    return res

def read_tag_events(scalar_tag, event_files, verbal=True, smooth_weight=0, interp=False, interp_x=None, meanstd=False):
    ''' read one tag from multiple events'''
    
    # 1. load events
    events = []
    for f in event_files:
        if verbal:
            print('start processing %s' % f)
        events.append(ea.EventAccumulator(f).Reload())
        
    # 2. read tag
    res = {}
    for i, event in enumerate(events):
        res[i] = read_scalar_tag(event, scalar_tag, smooth_weight=smooth_weight, interp=interp, interp_x=interp_x)
    
    # 3. calculate mean and std if needed
    if meanstd:
        assert(interp is True)
        vals = np.asarray([d[1] for d in res.values()])

    # 4. return
    return res if not meanstd else (res, np.mean(vals, axis=0), np.std(vals, axis=0))


def get_event_files(path, keywords=[], filterwords=[], recursive=False):
    '''extract event files under a path, selected by keywords and filtered by filterwords'''

    event_files = glob.glob('%s/**/events.out.tfevents*' % path, recursive=recursive)
    selected_files = [f for f in event_files if all([k in f for k in keywords]) and not any([k in f for k in filterwords])]
    return selected_files
