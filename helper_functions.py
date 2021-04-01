import cv2 as cv
import preprocessing_functions as pf
import datetime as datetime
import os
import cv2 as cv
import re
import numpy as np

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
  
def load_images_from_folder(folder, cutUruguay = True):
  '''
  Input:
  folder: path y nombre de la carpeta con las imagenes 
  numeric_name: borrar letras de los nombres de las imagenes al cargar a la lista
  cutUruguay: si True recorta la region de Uruguay al cargar 
  Output:
  images: lista con las imagenes cargadas
  time_stamp: lista datetime de las imagenes'''
  images = []
  time_stamp = []
  dia_ref = datetime.datetime(2019,12,31)
  for filename in os.listdir(folder):
      img = cv.imread(os.path.join(folder, filename))
      if img is not None:
          if (cutUruguay):
              images.append(img[67:185,109:237,0])
              
          else:
              images.append(img[:,:,0])
          
          img_name = re.sub("[^0-9]", "", filename)
          dt_image = dia_ref + datetime.timedelta(days=int(img_name[4:7]), hours =int(img_name[7:9]),
                      minutes = int(img_name[9:11]), seconds = int(img_name[11:]) )
          time_stamp.append(dt_image)
          

  return images,time_stamp