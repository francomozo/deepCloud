import numpy as np
import os
import re
import src.lib.preprocessing_functions as pf
import cv2 as cv
import datetime
from torch.utils.data import Dataset

def load_img(meta_path='data/meta',
             img_name='ART_2020020_111017.FR',
             mk_folder_path='data/C02-MK/2020',
             img_folder_path='data/C02-FR/2020'
    ):
   
    lats, lons = pf.read_meta(meta_path)
    
    dtime = pf.get_dtime(img_name)
    

    cosangs, cos_mask = pf.get_cosangs(dtime, lats, lons)
    img_mask = pf.load_mask(
      img_name, mk_folder_path, lats.size, lons.size
    )
    img = pf.load_img(
      img_name, img_folder_path, lats.size, lons.size
    )
    rimg = cv.inpaint(img, img_mask, 3, cv.INPAINT_NS)
    rp_image = pf.normalize(rimg, cosangs, 0.15)
    
    return rp_image   

def save_imgs_2npy(meta_path='data/meta',
            mk_folder_path='data/C02-MK/2020',
            img_folder_path='data/C02-FR/2020',
            destintation_path='data/images',
            split_days_into_folders=True
    ):
    """Saves images as Numpy arrays

    Args:
        meta_path (str, optional): Defaults to 'data/meta'.
        mk_folder_path (str, optional): Defaults to 'data/C02-MK/2020'.
        img_folder_path (str, optional): Defaults to 'data/C02-FR/2020'.
        destintation_path (str, optional): Defaults to 'data/images'.
        split_days_into_folders (bool, optional): Defaults to False.
    """

    for filename in os.listdir(img_folder_path):
        img = load_img(  # added needed arguments (franchesoni)
                    meta_path=meta_path,
                    img_name=filename,
                    mk_folder_path=mk_folder_path,
                    img_folder_path=img_folder_path,
        )
        img = np.asarray(img)

        if split_days_into_folders:
            day = re.sub("[^0-9]", "", filename)[4:7].lstrip("0")
            try:
                os.makedirs(os.path.join(os.getcwd(), destintation_path, "dia_" + day))
            except:
                pass
            path = os.path.join(destintation_path, "dia_" + day, os.path.splitext(filename)[0] + ".npy")
        
        else:
            try:
                os.makedirs(os.path.join(os.getcwd(), destintation_path, "loaded_images"))
            except:
                pass
            path = os.path.join(destintation_path, 'loaded_images', os.path.splitext(filename)[0] + ".npy")

        np.save(path, img)
        
        
def load_images_from_folder(folder, cutUruguay = True):
    """Loads images stored as Numpy arrays of day X to a list

    Args:
        folder (str): Where the images of day X are stored
        cutUruguay (bool, optional): Whether to crop to Uruguay. Defaults to True.

    Returns:
        images (list): List of Numpy arrays containing the images
        time_stamp (list): List with datetime of the images
    """
    
    images = []
    time_stamp = []
    dia_ref = datetime.datetime(2019,12,31)
    
    for filename in np.sort(os.listdir(folder)):
        img = np.load(os.path.join(folder, filename))
        
        if cutUruguay:
            images.append(img[67:185,109:237])
        else:
            images.append(img)
        
        img_name = re.sub("[^0-9]", "", filename)
        dt_image = dia_ref + datetime.timedelta(days=int(img_name[4:7]), hours =int(img_name[7:9]),
                    minutes = int(img_name[9:11]), seconds = int(img_name[11:]) )
        time_stamp.append(dt_image)

    return images, time_stamp

def load_by_batches(folder, current_imgs, time_stamp, list_size, last_img_filename, cutUruguay = True):
    """Loads the first "list_size" images from "folder" if "current_imgs"=[],
        if not, deletes the first element in the list, shift left on position, and
        reads the next image and time-stamp

    Args:
        folder (str): Where .npy arrays are stored
        current_imgs (list): Numpy arrays storing the images
        time_stamp ([type]): [description]
        list_size (int): Quantity of images to load , should be equal to the prediction horizon + 1
        cutUruguay (bool, optional): Slices image to keep only the region containing Uruguay. Defaults to True.
    """
    
    dia_ref = datetime.datetime(2019,12,31)
    sorted_img_list = np.sort(os.listdir(folder))
    
    if current_imgs == []:
        #for nth_img in range(list_size + 1):
        for nth_img in range(list_size ):
            filename = sorted_img_list[nth_img] # stores last img
            img = np.load(os.path.join(folder, filename))
            
            if cutUruguay:
                current_imgs.append(img[67:185,109:237])
            else:
                current_imgs.append(img)

            img_name = re.sub("[^0-9]", "", filename)
            dt_image = dia_ref + datetime.timedelta(days=int(img_name[4:7]), hours =int(img_name[7:9]),
                    minutes = int(img_name[9:11]), seconds = int(img_name[11:]) )
            time_stamp.append(dt_image)
    else:
        del current_imgs[0]
        del time_stamp[0]
        
        last_img_index = np.where(sorted_img_list == last_img_filename)[0][0]
        
        new_img_filename = sorted_img_list[last_img_index + 1]
        
        current_imgs.append(np.load(os.path.join(folder, new_img_filename)))
    
        img_name = re.sub("[^0-9]", "", new_img_filename)
        dt_image = dia_ref + datetime.timedelta(days=int(img_name[4:7]), hours =int(img_name[7:9]),
                    minutes = int(img_name[9:11]), seconds = int(img_name[11:]) )
        time_stamp.append(dt_image)
        
        filename = new_img_filename
    
    return current_imgs, time_stamp, filename

class SatelliteImagesDataset(Dataset):
    """ South America Satellite Images Dataset

    Args:
        root_dir (string): Directory with all images from day n.
        transform (callable, optional): Optional transform to be applied on a sample.
        
    Returns:
        [dict]: {'image': image, 'time_stamp': time_stamp}
    """

    dia_ref = datetime.datetime(2019,12,31)
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images_list = np.sort(os.listdir(self.root_dir))
        self.transform = transform
    
    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 
                                self.images_list[idx])
        image = np.load(img_name)
        if self.transform:
            image = self.transform(image)
        
        img_name = re.sub("[^0-9]", "", self.images_list[idx])
        time_stamp = self.dia_ref + datetime.timedelta(days=int(img_name[4:7]), hours =int(img_name[7:9]), 
                                                  minutes = int(img_name[9:11]), seconds = int(img_name[11:]) )
        
        sample = {'image': image,
                  'time_stamp': time_stamp}
        
        return sample
    
class CropImage(object):
    """ Whether to crop the images or not
    
    Args:
        limits (list): [x1, x2, y1, y2] where to crop the image.
    """

    def __init__(self, limits):
        self.x1, self.x2 = limits[0], limits[1]
        self.y1, self.y2 = limits[2], limits[3]
        
    def __call__(self, image):
        return image[self.y1:self.y2,self.x1:self.x2]
        