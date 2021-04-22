# USAGE:
#   Depracated functions that have usage.
#   Should be used as little as possible.
#

# First version of the SatelliteImagesDataset with sliding window. Takes too long.
# Loades images in each iteration.
class SatelliteImagesDatasetSW_v1(Dataset):
    """ [WARNING]: This function is depracated. Too slow. Use SatelliteImagesDatasetSW instead.
        South America Satellite Images Dataset

    Args:
        root_dir (string): Directory with all images from day n.
        window (int): Size of the moving window to load the images.
        transform (callable, optional): Optional transform to be applied on a sample.
        
        
    Returns:
        [dict]: {'images': images, 'time_stamps': time_stamps}
    """

    dia_ref = datetime.datetime(2019,12,31)
    
    
    
    def __init__(self, root_dir, window=1, transform=None):
        self.root_dir = root_dir
        self.images_list = np.sort(os.listdir(self.root_dir))
        self.transform = transform
        self.window = window
        
    
    def __len__(self):
        return len(self.images_list) - self.window + 1
    
    def __getitem__(self, idx):
        try:
            img_names = [os.path.join(self.root_dir, self.images_list[idx])
                         for idx in np.arange(idx, self.window + idx, 1)]

            images = np.array([np.load(img_name) for img_name in img_names])
            
            if self.transform:
                images = np.array([self.transform(image) for image in images])
                # images = np.array([self.transform(Image.fromarray(image)) for image in images])
            
            img_names = [re.sub("[^0-9]", "", self.images_list[idx]) 
                         for idx in np.arange(idx, self.window + idx, 1)]
           
            time_stamps = [self.dia_ref + datetime.timedelta(days=int(img_name[4:7]), 
                                                             hours=int(img_name[7:9]), 
                                                             minutes=int(img_name[9:11]), 
                                                             seconds=int(img_name[11:]))
                            for img_name in img_names]

            samples = {'images': images,
                      'time_stamps': [utils.datetime2str(ts) for ts in time_stamps]}
            
            return samples        
        except IndexError:
            print('End of sliding window')
            

class SatelliteImagesDataset(Dataset):
    """ [WARNING]: This function is depracated.
        First version of a simple dataset class for one image at a time.
        South America Satellite Images Dataset

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