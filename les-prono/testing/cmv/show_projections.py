from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np

def save_last_images_as_png():
  dir_of_imgs = Path('algorithms/cmv/predictions/projections')
  list_of_pimgs = [dir_of_imgs / name for name in
                  os.listdir(dir_of_imgs)
                  if not os.path.isdir(dir_of_imgs / name)]

  for pimg_path in list_of_pimgs:
    pimg = np.load(pimg_path)
    plt.figure()
    plt.imshow(pimg)
    plt.savefig(Path('testing/cmv/projections') / pimg_path.stem)

    print('max = ', np.nanmax(pimg))
    print('min = ', np.nanmin(pimg))

if __name__=='__main__':
  save_last_images_as_png()
