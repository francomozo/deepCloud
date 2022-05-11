# USAGE:
#   Custom preprocessing classes and functions
#

import numpy as np


class normalize_pixels(object):
    """ Normalizes images between [0,1] if mean0 is False, normalizes between [-0.5,0.5] if True
    """    
    def __init__(self, mean0=False):
        self.mean0 = mean0

    def __call__(self, in_frames, out_frames):
        if self.mean0:
            in_frames, out_frames = (in_frames/100)-0.5, (out_frames/100)-0.5
        else: 
            in_frames, out_frames = (in_frames/100), (out_frames/100)
        return in_frames, out_frames
        
class select_output_frame(object):
    """ 
    Function to select a specific frames when the output frames have multiple channels.
    """    
    def __init__(self, frame):
        self.frame = frame

    def __call__(self, in_frames, out_frames):
        out_frames = out_frames[self.frame, :, :]
        out_frames = np.expand_dims(out_frames, 0)
        return in_frames, out_frames   
    
