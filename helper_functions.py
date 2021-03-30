import cv2 as cv
import preprocessing_functions as pf

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