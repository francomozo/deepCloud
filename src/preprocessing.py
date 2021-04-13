# USAGE:
#   Custom preprocessing classes and functions
#

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