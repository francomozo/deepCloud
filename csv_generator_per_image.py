#csv_generator_per_image
import numpy as np
import os
import re
import csv
import argparse

from src import data
from src.lib import utils


def csv_generator_per_image(csv_path, img_folder_path, meta_path, region):
    filenames_days = sorted(os.listdir(img_folder_path))
    with open(csv_path, 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f)
        for filename_day in filenames_days:
            print(filename_day)
            day_folder_path = os.path.join(img_folder_path, filename_day)
            image_names = sorted(os.listdir(day_folder_path))
            for image_name in image_names:
                image_data = []
                image_data.append(image_name)
                _, cosangs_thresh = utils.get_cosangs_mask(meta_path=meta_path,
                                                   img_name=image_name)
                if region is None or region == 'mvd':
                    cosangs_img = cosangs_thresh[1550:1550+256, 1600:1600+256] # cut montevideo
                elif region == 'uru':
                    cosangs_img = cosangs_thresh[1205:1205+512, 1450:1450+512]
                elif region == 'region3':
                    cosangs_img = cosangs_thresh[800:800+1024, 1250:1250+1024]
                image_data.append(np.count_nonzero(cosangs_img == 1) / cosangs_img.size)
                writer.writerow(image_data)
      
ap = argparse.ArgumentParser(description='csv_generator_per_image')

ap.add_argument("--img-folder-path", default='/clusteruy/home03/DeepCloud/deepCloud/data/mvd/validation')
ap.add_argument("--region", default='mvd')
ap.add_argument("--csv-path", default='/clusteruy/home03/DeepCloud/deepCloud/data/mvd/val_cosangs_mvd.csv')
params = vars(ap.parse_args())

img_folder_path = params['img_folder_path']
region=params['region']
csv_path= params['csv_path']
meta_path='/clusteruy/home03/DeepCloud/deepCloud/data/raw/meta'

csv_generator_per_image(csv_path=csv_path, 
                        img_folder_path=img_folder_path, 
                        region=region, 
                        meta_path=meta_path)