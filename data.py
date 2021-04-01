import numpy as np
import os
import re
import preprocessing_functions as pf
import cv2 as cv
import datetime

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
        img = load_img(img_name=filename
        )
        img = np.asarray(img)

        if split_days_into_folders:
            day = re.sub("[^0-9]", "", filename)[4:7].lstrip("0")
            try:
                os.mkdir(destintation_path + "/dia_" + day)
            except:
                pass
            path = destintation_path + "/dia_" + day + "/" + os.path.splitext(filename)[0] + ".npy"
        
        else:
            try:
                os.mkdir(destintation_path + "/loaded_images")
            except:
                pass
            path = destintation_path + "/loaded_images/" + os.path.splitext(filename)[0] + ".png"

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
            images.append()
        
        img_name = re.sub("[^0-9]", "", filename)
        dt_image = dia_ref + datetime.timedelta(days=int(img_name[4:7]), hours =int(img_name[7:9]),
                    minutes = int(img_name[9:11]), seconds = int(img_name[11:]) )
        time_stamp.append(dt_image)

    return images, time_stamp

