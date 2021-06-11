# USAGE:
#   Custom preprocessing classes and functions
#

from torchvision.transforms import Resize
from PIL import Image
import torchvision.transforms.functional as F
import torch


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
    #def __init__(self):

    def __call__(self, in_frames,out_frames):
        #in_frames = torch.div(in_frames,100)
        #out_frames = torch.div(out_frames,100)
        in_frames,out_frames = in_frames/100 , out_frames/100
        return in_frames , out_frames
        
        
        
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