# USAGE:
#   Functions to load images, custom datasets, dataloaders and collate_fn,
#   train/val/test splits, date computations.
#

import numpy as np
import os
import re
import src.lib.preprocessing_functions as pf
import src.lib.utils as utils
import cv2 as cv
import datetime
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

class SatelliteImagesDatasetSW(Dataset):
    """ South America Satellite Images Dataset

    Args:
        root_dir (string): Directory with all images from day n.
        transform (callable, optional): Optional transform to be applied on a sample.
        
    Returns:
        [dict]: {'image': image, 'time_stamp': time_stamp}
    """

    dia_ref = datetime.datetime(2019,12,31)
    
    def __init__(self, root_dir, window=1, transform=None):
        self.root_dir = root_dir
        self.images_list = np.sort(os.listdir(self.root_dir))
        self.transform = transform
        self.window = window
    
    def __len__(self):
        return len(self.images_list) - self.window
    
    def __getitem__(self, idx):
        try:
            img_names = [os.path.join(self.root_dir, self.images_list[idx])
                         for idx in np.arange(idx, self.window + idx, 1)]

            images = np.array([np.load(img_name) for img_name in img_names])

            if self.transform:
                images = np.array([self.transform(image) for image in images])
            
            img_names = [re.sub("[^0-9]", "", self.images_list[idx]) 
                         for idx in np.arange(idx, self.window + idx, 1)]
           
            time_stamps = [self.dia_ref + datetime.timedelta(days=int(img_name[4:7]), 
                                                             hours=int(img_name[7:9]), 
                                                             minutes=int(img_name[9:11]), 
                                                             seconds = int(img_name[11:]))
                            for img_name in img_names]

            samples = {'images': images,
                      'time_stamps': [utils.datetime2str(ts) for ts in time_stamps]}
            
            return samples        
        except IndexError:
            print('End of sliding window')

def collate_fn_sw(batch):
    samples = default_collate(batch)
    samples['images'] = samples['images'].squeeze()
    samples['time_stamps'] = [''.join(ts) for ts in samples['time_stamps']]
    return samples

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
                  'time_stamp': utils.datetime2str(time_stamp)}
        
        return sample
    
def load_images_from_folder(folder, crop_region = 0):
    """Loads images stored as Numpy arrays of nth-day to a list

    Args:
        folder (str): Where the images of day X are stored
        crop_region (bool, optional): Regions 0(dont crop), 1, 2 and 3(Uruguay).

    Returns:
        images (list): List of Numpy arrays containing the images
        time_stamp (list): List with datetime of the images
    """
    
    current_imgs = []
    time_stamp = []
    dia_ref = datetime.datetime(2019,12,31)
    
    for filename in np.sort(os.listdir(folder)):
        img = np.load(os.path.join(folder, filename))
    
        if crop_region == 1:
            current_imgs.append(img[300:2500, 600:2800])
        elif crop_region == 2:
            current_imgs.append(img[500:2100, 800:2400])
        elif crop_region == 3:
            current_imgs.append(img[700:1700, 1200:2200])
        else:
            current_imgs.append(img)
        
        img_name = re.sub("[^0-9]", "", filename)
        dt_image = dia_ref + datetime.timedelta(days=int(img_name[4:7]), hours =int(img_name[7:9]),
                    minutes = int(img_name[9:11]), seconds = int(img_name[11:]) )
        time_stamp.append(dt_image)

    return current_imgs, time_stamp

def load_by_batches(folder, current_imgs, time_stamp, list_size, last_img_filename="", crop_region=0):
    """Loads the first "list_size" images from "folder" if "current_imgs"=[],
        if not, deletes the first element in the list, shift left on position, and
        reads the next image and time-stamp

    Args:
        folder (str): Where .npy arrays are stored
        current_imgs (list): Numpy arrays storing the images
        time_stamp ([type]): [description]
        list_size (int): Quantity of images to load , should be equal to the prediction horizon + 1
        crop_region (bool, optional): Regions 0(dont crop), 1, 2 and 3(Uruguay).
    """
    
    dia_ref = datetime.datetime(2019,12,31)
    sorted_img_list = np.sort(os.listdir(folder))
    
    if current_imgs == []:
        #for nth_img in range(list_size + 1):
        for nth_img in range(list_size ):
            filename = sorted_img_list[nth_img] # stores last img
            img = np.load(os.path.join(folder, filename))
                
            if crop_region == 1:
                current_imgs.append(img[300:2500, 600:2800])
            elif crop_region == 2:
                current_imgs.append(img[500:2100, 800:2400])
            elif crop_region == 3:
                current_imgs.append(img[700:1700, 1200:2200])
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

def load_img(meta_path='data/meta',
             img_name='ART_2020020_111017.FR',
             mk_folder_path='data/C02-MK/2020',
             img_folder_path='data/C02-FR/2020'
    ):
    """ Loads image from .FR .MK and metadata files into Numpy array

    Args:
        meta_path (str, optional): Defaults to 'data/meta'.
        img_name (str, optional): Defaults to 'ART_2020020_111017.FR'.
        mk_folder_path (str, optional): Defaults to 'data/C02-MK/2020'.
        img_folder_path (str, optional): Defaults to 'data/C02-FR/2020'.
    """
   
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
    """Saves images from "img_folder_path" to "destintation_path" as Numpy arrays
       (Uses load_img() function)

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
            path = os.path.join(destintation_path, "day_" + day, os.path.splitext(filename)[0] + ".npy")
        
        else:
            try:
                os.makedirs(os.path.join(os.getcwd(), destintation_path, "loaded_images"))
            except:
                pass
            path = os.path.join(destintation_path, 'loaded_images', os.path.splitext(filename)[0] + ".npy")

        np.save(path, img)
        
def save_imgs_list_2npy(imgs_list=[],
            meta_path='data/meta',
            mk_folder_path='data/C02-MK/2020',
            img_folder_path='data/C02-FR/2020',
            destintation_path='data/images',
            split_days_into_folders=True
    ):
    """Saves images as Numpy arrays to folders

    Args:
        imgs_list[] (list): List containing the names of the images to be saved. ie: days.
        meta_path (str, optional): Defaults to 'data/meta'.
        mk_folder_path (str, optional): Defaults to 'data/C02-MK/2020'.
        img_folder_path (str, optional): Defaults to 'data/C02-FR/2020'.
        destintation_path (str, optional): Defaults to 'data/images'.
        split_days_into_folders (bool, optional): Defaults to False.
    """

    for filename in imgs_list:
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

        