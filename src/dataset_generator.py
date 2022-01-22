import numpy as np
import os
import re

from src import data

fr_folder_path = '/clusteruy/home03/DeepCloud/deepCloud/data/raw_test/C02-FR'
mk_folder_path = '/clusteruy/home03/DeepCloud/deepCloud/data/raw_test/C02-MK'
meta_path = '/clusteruy/home03/DeepCloud/deepCloud/data/raw_test/meta'

def raw_test_2_datasets(imgs_list=[],
                        meta_path='data/meta',
                        mk_folder_path='data/C02-MK/2020',
                        img_folder_path='data/C02-FR/2020'
                        ):
    
    for filename in imgs_list:
        img = data.load_img(
                            meta_path=meta_path,
                            img_name=filename,
                            mk_folder_path=mk_folder_path,
                            img_folder_path=img_folder_path,
                        )
                        
        img = np.clip(img, 0, 100)
        
        regions = ['mvd', 'uru', 'region3']
        
        for region in regions:
            
            if region == 'mvd':
                img = img[1550: 1550 + 256, 1600: 1600 + 256]
            elif region == 'uru':
                img = img[1205: 1205 + 512, 1450: 1450 + 512]
            elif region == 'region3':           
                img = img[800: 800 + 1024, 1250: 1250 + 1024]

            day = re.sub("[^0-9]", "", filename)[:7]  # -> 2020XXX
            
            destintation_path = '/clusteruy/home03/DeepCloud/deepCloud/data/' + region + '/test/'
            
            try:
                os.makedirs(os.path.join(
                    os.getcwd(), destintation_path, day))
            except:
                pass
            path = os.path.join(destintation_path,
                                day, os.path.splitext(filename)[0] + ".npy")
            np.save(path, img)

if __name__ == '__main__':
    
    fr_filenames = sorted(os.listdir(fr_folder_path)):
    raw_test_2_datasets(imgs_list=fr_filenames,
                        meta_path=meta_path,
                        mk_folder_path=mk_folder_path,
                        img_folder_path=fr_folder_path
                        )
