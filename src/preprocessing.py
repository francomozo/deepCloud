# USAGE:
#   Custom preprocessing classes and functions
#

from torchvision.transforms import Resize
from PIL import Image
import torchvision.transforms.functional as F
import torch
import numpy as np


class CropImage(object):
    """ Whether to crop the images or not
    
    Args:
        crop_region (int):  1 - size=2200x2200.
                            2 - size=1600x1600.
                            3 - size=1000x1000 (Uruguay).
    """

    def __init__(self, crop_region):
        if crop_region == 1:
            self.x1, self.x2 = 300, 2500           
            self.y1, self.y2 = 600, 2800  
        elif crop_region == 2:
            self.x1, self.x2 = 500, 2100
            self.y1, self.y2 = 800, 2400
        elif crop_region == 3:
            self.x1, self.x2 = 700, 1700 
            self.y1, self.y2 = 1200, 2200
            
    def __call__(self, image):
        return image[self.x1:self.x2,self.y1:self.y2]
    
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
    

# class ResizeImage(object):
#     def __init__(self, size, interpolation=Image.BILINEAR, max_size=None):
#         self.size = size
#         self.interpolation = interpolation
#         self.max_size = max_size
    
#     def forward(self, img):
#         """
#         Args:
#             img (Numpy ndarray): Image to be scaled.
#         Returns:
#             PIL Image or Tensor: Rescaled image.
#         """
#         return F.resize(Image.fromarray(img), self.size, self.interpolation)    
    
#     def __repr__(self):
#         interpolate_str = self.interpolation.value
#         return self.__class__.__name__ + '(size={0}, interpolation={1}, max_size={2})'.format(
#             self.size, interpolate_str, self.max_size)
        
#     def __call__(self, img):
#         return self.forward(img)